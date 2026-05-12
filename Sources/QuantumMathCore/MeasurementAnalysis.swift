import Foundation

public enum MeasurementState: Hashable, Sendable, Codable {
    case pure(Ket)
    case density(Operator)

    public var space: Space {
        switch self {
        case let .pure(ket):
            return ket.space
        case let .density(op):
            return op.domain
        }
    }

    public var basis: Basis {
        switch self {
        case let .pure(ket):
            return ket.basis
        case let .density(op):
            return op.columnBasis
        }
    }
}

public struct ProjectiveMeasurement: Hashable, Sendable, Codable {
    public let observable: Operator

    public init(observable: Operator, config: QuantumMathConfig) throws {
        guard observable.domain == observable.codomain,
              observable.rowBasis == observable.columnBasis else {
            throw QuantumMathError.operationNotDefined(
                "Projective measurement requires square observable with matching basis."
            )
        }
        guard observable.entries.isHermitian(tolerance: config.matrixTraitEpsilon) else {
            throw QuantumMathError.operationNotDefined("Observable must be Hermitian.")
        }
        self.observable = observable
    }
}

public struct MeasurementOutcome: Hashable, Sendable, Codable {
    public let eigenvalue: Scalar
    public let multiplicity: Int
    public let projector: Operator
    public let probability: Scalar
    public let postMeasurementState: MeasurementState?
    public let isZeroProbability: Bool
}

public struct MeasurementMoments: Hashable, Sendable, Codable {
    public let expectedValue: Scalar
    public let expectedSquaredValue: Scalar
    public let variance: Scalar
    public let standardDeviation: Scalar
}

public struct MeasurementPosterior: Hashable, Sendable, Codable {
    public let state: Operator
    public let purity: Scalar
}

public struct MeasurementAnalysisResult: Hashable, Sendable, Codable {
    public let observable: ProjectiveMeasurement
    public let state: MeasurementState
    public let outcomes: [MeasurementOutcome]
    public let probabilityTotal: Scalar
    public let moments: MeasurementMoments
    public let posterior: MeasurementPosterior
}

public struct MeasurementAnalyzer {
    public let config: QuantumMathConfig
    public let spectralAnalyzer: SpectralAnalyzer

    public init(
        config: QuantumMathConfig,
        spectralAnalyzer: SpectralAnalyzer
    ) {
        self.config = config
        self.spectralAnalyzer = spectralAnalyzer
    }

    public func analyze(
        observable: Operator,
        state: MeasurementState
    ) throws -> MeasurementAnalysisResult {
        let measurement = try ProjectiveMeasurement(observable: observable, config: config)
        guard measurement.observable.domain.isCoordinateCompatible(with: state.space),
              measurement.observable.columnBasis.isCoordinateCompatible(with: state.basis) else {
            throw QuantumMathError.operationNotDefined("Measurement basis or space mismatch.")
        }
        let analysisState = try validatedState(
            alignedState(state, to: measurement.observable.columnBasis)
        )

        let spectral = try spectralAnalyzer.hermitianDecomposition(measurement.observable)
        let outcomes = try spectral.decomposition.components.map { component in
            let projector = try projector(
                for: component,
                space: measurement.observable.domain,
                basis: measurement.observable.columnBasis
            )
            let probability = try validProbability(rawProbability(for: projector, state: analysisState))
            let isZeroProbability = probability.approximateValue.re <= config.scalarComparisonEpsilon
            let post = isZeroProbability ? nil : try postMeasurementState(
                state: analysisState,
                projector: projector,
                probability: probability
            )
            return MeasurementOutcome(
                eigenvalue: component.eigenvalue,
                multiplicity: component.multiplicity,
                projector: projector,
                probability: probability,
                postMeasurementState: post,
                isZeroProbability: isZeroProbability
            )
        }

        let probabilityTotal = outcomes.map(\.probability).reduce(.zero, +)
        guard probabilityTotal.approximatelyEquals(.one, epsilon: config.spectralResidualThreshold) else {
            throw QuantumMathError.operationNotDefined("Measurement probabilities must sum to one.")
        }

        let moments = try makeMoments(outcomes: outcomes)
        let posterior = try nonselectivePosterior(state: analysisState, projectors: outcomes.map(\.projector))

        return MeasurementAnalysisResult(
            observable: measurement,
            state: analysisState,
            outcomes: outcomes,
            probabilityTotal: probabilityTotal,
            moments: moments,
            posterior: posterior
        )
    }

    private func alignedState(_ state: MeasurementState, to basis: Basis) throws -> MeasurementState {
        switch state {
        case let .pure(ket):
            guard ket.basis != basis else {
                return state
            }
            return .pure(try Ket(space: basis.space, basis: basis, coefficients: ket.coefficients))
        case let .density(operatorValue):
            guard operatorValue.columnBasis != basis || operatorValue.rowBasis != basis else {
                return state
            }
            return .density(try Operator(
                domain: basis.space,
                codomain: basis.space,
                columnBasis: basis,
                rowBasis: basis,
                entries: operatorValue.entries
            ))
        }
    }

    private func validatedState(_ state: MeasurementState) throws -> MeasurementState {
        switch state {
        case let .pure(ket):
            let normSquared = QuantumDomain.normSquared(ket)
            guard abs(normSquared - 1) <= config.scalarComparisonEpsilon else {
                throw QuantumMathError.operationNotDefined(
                    "Measurement requires a normalized ket. Insert an explicit normalize line first."
                )
            }
            return state
        case let .density(operatorValue):
            guard isTraceOneHermitianState(operatorValue) else {
                throw QuantumMathError.operationNotDefined(
                    "Measurement state must be a normalized ket or trace-one density operator."
                )
            }
            return state
        }
    }

    private func isTraceOneHermitianState(_ operatorValue: Operator) -> Bool {
        guard operatorValue.domain == operatorValue.codomain,
              operatorValue.rowBasis == operatorValue.columnBasis,
              operatorValue.entries.isHermitian(tolerance: config.matrixTraitEpsilon),
              let trace = operatorValue.entries.trace() else {
            return false
        }
        return trace.approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon)
    }

    private func projector(for component: EigenComponent, space: Space, basis: Basis) throws -> Operator {
        let entries = try component.eigenvectors.reduce(
            Matrix<Scalar>.zero(rows: space.dimension, cols: space.dimension)
        ) { partial, eigenvector in
            let projector = try QuantumDomain.outerProduct(eigenvector, try QuantumDomain.dagger(eigenvector))
            let values = zip(partial.values, projector.entries.values).map(+)
            return Matrix<Scalar>(uncheckedRows: partial.rows, cols: partial.cols, values: values)
        }

        return try Operator(
            domain: space,
            codomain: space,
            columnBasis: basis,
            rowBasis: basis,
            entries: entries
        )
    }

    private func rawProbability(for projector: Operator, state: MeasurementState) throws -> Scalar {
        switch state {
        case let .pure(ket):
            let projected = try QuantumDomain.applyOperator(projector, ket)
            return try QuantumDomain.innerProduct(projected, projected)
        case let .density(density):
            let projected = try QuantumDomain.compose(
                try QuantumDomain.compose(projector, density),
                projector
            )
            return trace(of: projected.entries)
        }
    }

    private func validProbability(_ probability: Scalar) throws -> Scalar {
        let approximate = probability.approximateValue
        guard abs(approximate.im) <= config.spectralResidualThreshold else {
            throw QuantumMathError.operationNotDefined("Measurement probability must be real.")
        }
        guard approximate.re >= -config.spectralResidualThreshold else {
            throw QuantumMathError.operationNotDefined("Measurement probability must be non-negative.")
        }
        return .approx(ComplexNumber(re: max(approximate.re, 0), im: 0))
    }

    private func postMeasurementState(
        state: MeasurementState,
        projector: Operator,
        probability: Scalar
    ) throws -> MeasurementState {
        switch state {
        case let .pure(ket):
            let projected = try QuantumDomain.applyOperator(projector, ket)
            if sameProjectiveRay(projected, ket, epsilon: config.spectralResidualThreshold) {
                return .pure(ket)
            }
            let normSquared = QuantumDomain.normSquared(projected)
            guard normSquared > config.scalarComparisonEpsilon else {
                return .pure(ket)
            }
            let scale = 1 / Foundation.sqrt(normSquared)
            let normalized = try Ket(
                space: projected.space,
                basis: projected.basis,
                coefficients: projected.coefficients.map {
                    .approx($0.approximateValue.scaled(by: scale))
                }
            )
            return .pure(normalized)
        case let .density(density):
            let numerator = try QuantumDomain.compose(
                try QuantumDomain.compose(projector, density),
                projector
            )
            guard let normalized = divide(operatorValue: numerator, by: probability) else {
                return .density(density)
            }
            return .density(normalized)
        }
    }

    private func sameProjectiveRay(
        _ lhs: Ket,
        _ rhs: Ket,
        epsilon: Double
    ) -> Bool {
        guard lhs.space.isCoordinateCompatible(with: rhs.space),
              lhs.basis == rhs.basis,
              lhs.coefficients.count == rhs.coefficients.count else {
            return false
        }

        let lhsNormSquared = lhs.coefficients.reduce(0.0) { partial, scalar in
            partial + scalar.magnitudeSquaredApproximate
        }
        let rhsNormSquared = rhs.coefficients.reduce(0.0) { partial, scalar in
            partial + scalar.magnitudeSquaredApproximate
        }
        guard lhsNormSquared > config.scalarComparisonEpsilon,
              rhsNormSquared > config.scalarComparisonEpsilon else {
            return false
        }

        let inner = zip(lhs.coefficients, rhs.coefficients).reduce(
            ComplexNumber(re: 0, im: 0)
        ) { partial, pair in
            partial + (pair.0.approximateValue.conjugated * pair.1.approximateValue)
        }
        let fidelity = inner.magnitudeSquared / (lhsNormSquared * rhsNormSquared)
        return abs(1 - fidelity) <= epsilon
    }

    private func nonselectivePosterior(
        state: MeasurementState,
        projectors: [Operator]
    ) throws -> MeasurementPosterior {
        let density = try densityOperator(from: state)
        let accumulated = try projectors.reduce(
            Matrix<Scalar>.zero(rows: density.entries.rows, cols: density.entries.cols)
        ) { partial, projector in
            let projected = try QuantumDomain.compose(
                try QuantumDomain.compose(projector, density),
                projector
            )
            let values = zip(partial.values, projected.entries.values).map(+)
            return Matrix<Scalar>(uncheckedRows: partial.rows, cols: partial.cols, values: values)
        }

        let stateOperator = try Operator(
            domain: density.domain,
            codomain: density.codomain,
            columnBasis: density.columnBasis,
            rowBasis: density.rowBasis,
            entries: accumulated
        )
        let squared = try QuantumDomain.compose(stateOperator, stateOperator)
        let purity = trace(of: squared.entries)

        return MeasurementPosterior(state: stateOperator, purity: purity)
    }

    private func makeMoments(outcomes: [MeasurementOutcome]) throws -> MeasurementMoments {
        let expected = outcomes.reduce(.zero) { partial, outcome in
            partial + (outcome.eigenvalue * outcome.probability)
        }
        let expectedSquared = outcomes.reduce(.zero) { partial, outcome in
            partial + (outcome.eigenvalue * outcome.eigenvalue * outcome.probability)
        }
        let variance = expectedSquared - (expected * expected)
        let varianceReal = max(variance.approximateValue.re, 0)
        let std = Scalar.approx(
            ComplexNumber(re: Foundation.sqrt(varianceReal), im: 0)
        )
        return MeasurementMoments(
            expectedValue: expected,
            expectedSquaredValue: expectedSquared,
            variance: .approx(ComplexNumber(re: varianceReal, im: 0)),
            standardDeviation: std
        )
    }

    private func densityOperator(from state: MeasurementState) throws -> Operator {
        switch state {
        case let .density(op):
            return op
        case let .pure(ket):
            return try QuantumDomain.outerProduct(ket, try QuantumDomain.dagger(ket))
        }
    }

    private func divide(operatorValue: Operator, by scalar: Scalar) -> Operator? {
        guard let reciprocal = Scalar.one.divided(by: scalar) else {
            return nil
        }
        let values = operatorValue.entries.values.map { $0 * reciprocal }
        let matrix = Matrix<Scalar>(
            uncheckedRows: operatorValue.entries.rows,
            cols: operatorValue.entries.cols,
            values: values
        )
        return try? Operator(
            domain: operatorValue.domain,
            codomain: operatorValue.codomain,
            columnBasis: operatorValue.columnBasis,
            rowBasis: operatorValue.rowBasis,
            entries: matrix
        )
    }

    private func trace(of matrix: Matrix<Scalar>) -> Scalar {
        guard matrix.rows == matrix.cols else {
            return .approx(ComplexNumber(re: .nan, im: .nan))
        }
        return (0..<matrix.rows).reduce(.zero) { partial, index in
            partial + matrix[index, index]
        }
    }

}
