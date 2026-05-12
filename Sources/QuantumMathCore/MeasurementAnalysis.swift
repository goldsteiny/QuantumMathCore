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

        let spectral = try spectralAnalyzer.hermitianDecomposition(measurement.observable)
        let outcomes = try spectral.decomposition.components.map { component in
            let projector = try projector(for: component, space: state.space, basis: state.basis)
            let probability = try validProbability(rawProbability(for: projector, state: state))
            let isZeroProbability = probability.approximateValue.re <= config.scalarComparisonEpsilon
            let post = isZeroProbability ? nil : try postMeasurementState(
                state: state,
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
        let posterior = try nonselectivePosterior(state: state, projectors: outcomes.map(\.projector))

        return MeasurementAnalysisResult(
            observable: measurement,
            state: state,
            outcomes: outcomes,
            probabilityTotal: probabilityTotal,
            moments: moments,
            posterior: posterior
        )
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
            entries: exactifiedMatrix(entries, label: "measurement.projector")
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
        return ScalarExactificationAdapter.exactify(
            .approx(ComplexNumber(re: max(approximate.re, 0), im: 0)),
            label: "measurement.probability",
            config: config
        ).scalar
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
            return .pure(
                StateVectorExactificationAdapter.exactify(
                    normalized,
                    label: "measurement.post_state",
                    config: config
                ).ket
            )
        case let .density(density):
            let numerator = try QuantumDomain.compose(
                try QuantumDomain.compose(projector, density),
                projector
            )
            guard let normalized = divide(operatorValue: numerator, by: probability) else {
                return .density(density)
            }
            return .density(try exactifiedOperator(normalized, label: "measurement.post_density"))
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

        let stateOperator = try exactifiedOperator(Operator(
            domain: density.domain,
            codomain: density.codomain,
            columnBasis: density.columnBasis,
            rowBasis: density.rowBasis,
            entries: accumulated
        ), label: "measurement.posterior")
        let squared = try QuantumDomain.compose(stateOperator, stateOperator)
        let purity = ScalarExactificationAdapter.exactify(
            trace(of: squared.entries),
            label: "measurement.posterior.purity",
            config: config
        ).scalar

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
            expectedValue: ScalarExactificationAdapter.exactify(
                expected,
                label: "measurement.expected",
                config: config
            ).scalar,
            expectedSquaredValue: ScalarExactificationAdapter.exactify(
                expectedSquared,
                label: "measurement.expected_squared",
                config: config
            ).scalar,
            variance: ScalarExactificationAdapter.exactify(
                .approx(ComplexNumber(re: varianceReal, im: 0)),
                label: "measurement.variance",
                config: config
            ).scalar,
            standardDeviation: ScalarExactificationAdapter.exactify(
                std,
                label: "measurement.stddev",
                config: config
            ).scalar
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

    private func exactifiedOperator(_ operatorValue: Operator, label: String) throws -> Operator {
        try Operator(
            domain: operatorValue.domain,
            codomain: operatorValue.codomain,
            columnBasis: operatorValue.columnBasis,
            rowBasis: operatorValue.rowBasis,
            entries: exactifiedMatrix(operatorValue.entries, label: label)
        )
    }

    private func exactifiedMatrix(_ matrix: Matrix<Scalar>, label: String) -> Matrix<Scalar> {
        Matrix<Scalar>(
            uncheckedRows: matrix.rows,
            cols: matrix.cols,
            values: matrix.values.enumerated().map { index, scalar in
                ScalarExactificationAdapter.exactify(
                    scalar,
                    label: "\(label).m\(index)",
                    config: config
                ).scalar
            }
        )
    }
}
