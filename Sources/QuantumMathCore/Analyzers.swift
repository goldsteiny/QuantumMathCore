import Foundation
import ExactValueRecoveryCore

public struct ExactificationMetadata: Hashable, Sendable, Codable {
    public enum Status: Hashable, Sendable, Codable {
        case verifiedExact
        case approximateRetained
        case unresolved(UnresolvedReason)
    }

    public let status: Status
    public let witness: QuantumVerificationWitness?

    public init(status: Status, witness: QuantumVerificationWitness?) {
        self.status = status
        self.witness = witness
    }
}

public struct QuantumMathBackends: Sendable {
    public let hermitianEigenSolver: any HermitianEigenSolver
    public let singularValueSolver: any SingularValueSolver

    public init(
        hermitianEigenSolver: any HermitianEigenSolver,
        singularValueSolver: any SingularValueSolver
    ) {
        self.hermitianEigenSolver = hermitianEigenSolver
        self.singularValueSolver = singularValueSolver
    }
}

public struct SpectralAnalysisResult: Hashable, Sendable, Codable {
    public let decomposition: SpectralDecomposition
    public let exactificationMetadataByComponent: [Int: ExactificationMetadata]
    public let vectorExactificationMetadataByComponent: [Int: [String: ExactificationMetadata]]

    public init(
        decomposition: SpectralDecomposition,
        exactificationMetadataByComponent: [Int: ExactificationMetadata],
        vectorExactificationMetadataByComponent: [Int: [String: ExactificationMetadata]] = [:]
    ) {
        self.decomposition = decomposition
        self.exactificationMetadataByComponent = exactificationMetadataByComponent
        self.vectorExactificationMetadataByComponent = vectorExactificationMetadataByComponent
    }
}

public struct SingularValueAnalysisResult: Hashable, Sendable, Codable {
    public let decomposition: SingularValueDecomposition
    public let exactificationMetadataByComponent: [Int: ExactificationMetadata]
    public let leftVectorExactificationMetadataByComponent: [Int: [Int: ExactificationMetadata]]
    public let rightVectorExactificationMetadataByComponent: [Int: [Int: ExactificationMetadata]]

    public init(
        decomposition: SingularValueDecomposition,
        exactificationMetadataByComponent: [Int: ExactificationMetadata],
        leftVectorExactificationMetadataByComponent: [Int: [Int: ExactificationMetadata]] = [:],
        rightVectorExactificationMetadataByComponent: [Int: [Int: ExactificationMetadata]] = [:]
    ) {
        self.decomposition = decomposition
        self.exactificationMetadataByComponent = exactificationMetadataByComponent
        self.leftVectorExactificationMetadataByComponent = leftVectorExactificationMetadataByComponent
        self.rightVectorExactificationMetadataByComponent = rightVectorExactificationMetadataByComponent
    }
}

public struct SpectralAnalyzer {
    public let config: QuantumMathConfig
    public let backend: any HermitianEigenSolver

    public init(config: QuantumMathConfig, backend: any HermitianEigenSolver) {
        self.config = config
        self.backend = backend
    }

    public func hermitianDecomposition(
        _ operatorValue: Operator
    ) throws -> SpectralAnalysisResult {
        guard operatorValue.domain == operatorValue.codomain,
              operatorValue.rowBasis == operatorValue.columnBasis else {
            throw QuantumMathError.operationNotDefined(
                "Hermitian spectral decomposition requires a square operator with matching basis."
            )
        }

        guard operatorValue.entries.isHermitian(tolerance: config.matrixTraitEpsilon) else {
            throw QuantumMathError.operationNotDefined(
                "Hermitian spectral decomposition requires a Hermitian operator."
            )
        }

        let raw = try backend.decompose(operatorValue.entries)
        let decomposition = try makeApproximateDecomposition(
            raw: raw,
            space: operatorValue.domain,
            basis: operatorValue.columnBasis
        )

        var metadata: [Int: ExactificationMetadata] = [:]
        var vectorMetadata: [Int: [String: ExactificationMetadata]] = [:]
        let updatedComponents = decomposition.components.enumerated().map { index, component in
            let exactified = ScalarExactificationAdapter.exactify(
                component.eigenvalue,
                label: "spectral.\(index).eigenvalue",
                config: config
            )
            metadata[index] = metadataFor(outcome: exactified.outcome)

            let exactifiedVectors = component.eigenvectors.enumerated().map { vectorIndex, eigenvector in
                let exactifiedVector = StateVectorExactificationAdapter.exactify(
                    eigenvector,
                    label: "spectral.\(index).v\(vectorIndex)",
                    config: config
                )
                for (coefficientIndex, coefficientMetadata) in exactifiedVector.metadataByCoefficient {
                    vectorMetadata[index, default: [:]]["v\(vectorIndex).c\(coefficientIndex)"] = coefficientMetadata
                }
                return exactifiedVector.ket
            }

            return EigenComponent(
                eigenvalue: exactified.scalar,
                multiplicity: component.multiplicity,
                eigenvectors: exactifiedVectors
            )
        }

        return SpectralAnalysisResult(
            decomposition: SpectralDecomposition(
                space: decomposition.space,
                basis: decomposition.basis,
                components: updatedComponents
            ),
            exactificationMetadataByComponent: metadata,
            vectorExactificationMetadataByComponent: vectorMetadata
        )
    }

    private func makeApproximateDecomposition(
        raw: HermitianEigenDecompositionRaw,
        space: Space,
        basis: Basis
    ) throws -> SpectralDecomposition {
        guard raw.dimension == space.dimension else {
            throw QuantumMathError.backendFailure(
                BackendFailure(
                    backend: "HermitianEigenSolver",
                    kind: .invalidInput,
                    message: "Backend returned dimension \(raw.dimension) but space dimension is \(space.dimension)."
                )
            )
        }

        let pairs: [(Scalar, Ket)] = try (0..<raw.dimension).map { column in
            let vector = phaseCanonicalized(raw.eigenvector(at: column))
            guard vector.allSatisfy({ $0.isFinite }) else {
                throw QuantumMathError.backendFailure(
                    BackendFailure(
                        backend: "HermitianEigenSolver",
                        kind: .nonFiniteOutput,
                        message: "Eigenvector output contained non-finite values."
                    )
                )
            }

            let ket = try Ket(
                space: space,
                basis: basis,
                coefficients: vector.map { .approx(sanitized($0)) }
            )
            let eigenvalue = Scalar.approx(
                ComplexNumber(
                    re: sanitizedReal(raw.eigenvalues[column]),
                    im: 0
                )
            )
            return (eigenvalue, ket)
        }
        .sorted { lhs, rhs in
            lhs.0.approximateValue.re < rhs.0.approximateValue.re
        }

        var components: [EigenComponent] = []
        for pair in pairs {
            if let lastIndex = components.indices.last,
               components[lastIndex].eigenvalue.approximatelyEquals(
                   pair.0,
                   epsilon: config.matrixTraitEpsilon
               ) {
                let last = components[lastIndex]
                components[lastIndex] = EigenComponent(
                    eigenvalue: last.eigenvalue,
                    multiplicity: last.multiplicity + 1,
                    eigenvectors: last.eigenvectors + [pair.1]
                )
            } else {
                components.append(
                    EigenComponent(
                        eigenvalue: pair.0,
                        multiplicity: 1,
                        eigenvectors: [pair.1]
                    )
                )
            }
        }

        return SpectralDecomposition(
            space: space,
            basis: basis,
            components: components
        )
    }
}

public struct SingularValueAnalyzer {
    public let config: QuantumMathConfig
    public let backend: any SingularValueSolver

    public init(config: QuantumMathConfig, backend: any SingularValueSolver) {
        self.config = config
        self.backend = backend
    }

    public func decompose(_ operatorValue: Operator) throws -> SingularValueAnalysisResult {
        guard operatorValue.entries.rows <= config.maxComputableDimension,
              operatorValue.entries.cols <= config.maxComputableDimension else {
            throw QuantumMathError.operationNotDefined(
                "SVD exceeds configured dimension limit \(config.maxComputableDimension)."
            )
        }

        let raw = try backend.decompose(operatorValue.entries)
        guard raw.rows == operatorValue.entries.rows,
              raw.cols == operatorValue.entries.cols else {
            throw QuantumMathError.backendFailure(
                BackendFailure(
                    backend: "SingularValueSolver",
                    kind: .invalidInput,
                    message: "Backend returned shape \(raw.rows)x\(raw.cols) for input \(operatorValue.entries.rows)x\(operatorValue.entries.cols)."
                )
            )
        }

        var metadata: [Int: ExactificationMetadata] = [:]
        var leftVectorMetadata: [Int: [Int: ExactificationMetadata]] = [:]
        var rightVectorMetadata: [Int: [Int: ExactificationMetadata]] = [:]
        let components = try (0..<raw.rank).map { index in
            var left = raw.leftVector(at: index)
            var right = raw.rightVector(at: index)
            phaseCanonicalize(left: &left, right: &right)

            let approximateLeftVector = try Ket(
                space: operatorValue.codomain,
                basis: operatorValue.rowBasis,
                coefficients: left.map { .approx(sanitized($0)) }
            )
            let approximateRightVector = try Ket(
                space: operatorValue.domain,
                basis: operatorValue.columnBasis,
                coefficients: right.map { .approx(sanitized($0)) }
            )
            let leftExactification = StateVectorExactificationAdapter.exactify(
                approximateLeftVector,
                label: "svd.\(index).left",
                config: config
            )
            let rightExactification = StateVectorExactificationAdapter.exactify(
                approximateRightVector,
                label: "svd.\(index).right",
                config: config
            )
            leftVectorMetadata[index] = leftExactification.metadataByCoefficient
            rightVectorMetadata[index] = rightExactification.metadataByCoefficient

            let singularValue = Scalar.approx(
                ComplexNumber(
                    re: sanitizedNonNegative(raw.singularValues[index]),
                    im: 0
                )
            )
            let exactified = ScalarExactificationAdapter.exactify(
                singularValue,
                label: "svd.\(index).sigma",
                config: config
            )
            metadata[index] = metadataFor(outcome: exactified.outcome)

            return SingularValueComponent(
                singularValue: exactified.scalar,
                leftVector: leftExactification.ket,
                rightVector: rightExactification.ket
            )
        }

        return SingularValueAnalysisResult(
            decomposition: SingularValueDecomposition(
                domain: operatorValue.domain,
                codomain: operatorValue.codomain,
                columnBasis: operatorValue.columnBasis,
                rowBasis: operatorValue.rowBasis,
                components: components
            ),
            exactificationMetadataByComponent: metadata,
            leftVectorExactificationMetadataByComponent: leftVectorMetadata,
            rightVectorExactificationMetadataByComponent: rightVectorMetadata
        )
    }
}

private func metadataFor(
    outcome: ExactificationOutcome<QuantumVerificationWitness>
) -> ExactificationMetadata {
    switch outcome {
    case let .exact(_, witness):
        return ExactificationMetadata(status: .verifiedExact, witness: witness)
    case let .mixed(_, witness, _):
        return ExactificationMetadata(status: .verifiedExact, witness: witness)
    case let .unresolved(reason):
        return ExactificationMetadata(status: .unresolved(reason), witness: nil)
    }
}

private func phaseCanonicalized(_ vector: [ComplexNumber]) -> [ComplexNumber] {
    guard let pivotIndex = vector.indices.max(by: { lhs, rhs in
        vector[lhs].magnitudeSquared < vector[rhs].magnitudeSquared
    }) else {
        return vector
    }

    let pivot = vector[pivotIndex]
    let magnitude = Foundation.sqrt(pivot.magnitudeSquared)
    guard magnitude > 0 else {
        return vector
    }

    let inversePhase = ComplexNumber(
        re: pivot.re / magnitude,
        im: -pivot.im / magnitude
    )
    return vector.map { sanitized($0 * inversePhase) }
}

private func phaseCanonicalize(left: inout [ComplexNumber], right: inout [ComplexNumber]) {
    let pivot = canonicalPhasePivot(in: right) ?? canonicalPhasePivot(in: left)
    guard let pivot else {
        return
    }

    let magnitude = Foundation.sqrt(pivot.magnitudeSquared)
    guard magnitude > 0 else {
        return
    }

    let inversePhase = ComplexNumber(
        re: pivot.re / magnitude,
        im: -pivot.im / magnitude
    )
    left = left.map { sanitized($0 * inversePhase) }
    right = right.map { sanitized($0 * inversePhase) }
}

private func canonicalPhasePivot(in vector: [ComplexNumber]) -> ComplexNumber? {
    guard let pivotIndex = vector.indices.max(by: { lhs, rhs in
        vector[lhs].magnitudeSquared < vector[rhs].magnitudeSquared
    }) else {
        return nil
    }
    return vector[pivotIndex]
}

private func sanitized(_ value: ComplexNumber) -> ComplexNumber {
    ComplexNumber(
        re: value.re == 0 ? 0 : value.re,
        im: value.im == 0 ? 0 : value.im
    )
}

private func sanitizedReal(_ value: Double) -> Double {
    abs(value) <= 1e-12 ? 0 : value
}

private func sanitizedNonNegative(_ value: Double) -> Double {
    if !value.isFinite {
        return value
    }
    if abs(value) <= 1e-12 {
        return 0
    }
    return max(0, value)
}
