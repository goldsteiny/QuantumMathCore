import Foundation

public enum EntanglementClassification: String, Hashable, Sendable, Codable {
    case product
    case entangled
    case maximallyEntangled
}

public struct BipartitePureStateAnalysis: Hashable, Sendable, Codable {
    public let state: Ket
    public let leftDimension: Int
    public let rightDimension: Int
    public let schmidtRank: Int
    public let schmidtCoefficients: [Scalar]
    public let classification: EntanglementClassification

    public init(
        state: Ket,
        leftDimension: Int,
        rightDimension: Int,
        schmidtRank: Int,
        schmidtCoefficients: [Scalar],
        classification: EntanglementClassification
    ) {
        self.state = state
        self.leftDimension = leftDimension
        self.rightDimension = rightDimension
        self.schmidtRank = schmidtRank
        self.schmidtCoefficients = schmidtCoefficients
        self.classification = classification
    }
}

public struct EntanglementAnalyzer: Sendable {
    public let config: QuantumMathConfig
    public let backend: any SingularValueSolver

    public init(config: QuantumMathConfig, backend: any SingularValueSolver) {
        self.config = config
        self.backend = backend
    }

    public func analyzeBipartitePureState(_ state: Ket) throws -> BipartitePureStateAnalysis {
        guard state.space.factors.count == 2 else {
            throw QuantumMathError.operationNotDefined(
                "Bipartite entanglement analysis requires a pure ket on exactly two tensor factors."
            )
        }
        guard state.basis.isComputationalCoordinateBasis else {
            throw QuantumMathError.operationNotDefined(
                "Bipartite entanglement analysis requires computational-coordinate basis."
            )
        }
        guard abs(QuantumDomain.normSquared(state) - 1) <= config.scalarComparisonEpsilon else {
            throw QuantumMathError.operationNotDefined("Bipartite pure state must be normalized.")
        }

        let leftDimension = state.space.factors[0].dimension
        let rightDimension = state.space.factors[1].dimension
        let coefficientMatrix = try Matrix(
            rows: leftDimension,
            cols: rightDimension,
            values: state.coefficients
        )
        let raw = try backend.decompose(coefficientMatrix)
        let coefficients = raw.singularValues.map {
            Scalar.approx(ComplexNumber(re: max($0, 0), im: 0))
        }
        let significant = coefficients.filter {
            $0.approximateValue.re > config.scalarComparisonEpsilon
        }
        let schmidtRank = significant.count

        let classification: EntanglementClassification
        if schmidtRank <= 1 {
            classification = .product
        } else if leftDimension == rightDimension,
                  schmidtRank == leftDimension,
                  significant.allSatisfy({
                      abs($0.approximateValue.re - (1 / Foundation.sqrt(Double(leftDimension)))) <= config.scalarComparisonEpsilon
                  }) {
            classification = .maximallyEntangled
        } else {
            classification = .entangled
        }

        return BipartitePureStateAnalysis(
            state: state,
            leftDimension: leftDimension,
            rightDimension: rightDimension,
            schmidtRank: schmidtRank,
            schmidtCoefficients: coefficients,
            classification: classification
        )
    }
}

public struct MaximallyEntangledResource: Hashable, Sendable, Codable {
    public let state: Ket
    public let dimension: Int
    public let analysis: BipartitePureStateAnalysis
    public let factorOrder: [CommunicationFactorRole]

    public init(
        state: Ket,
        analysis: BipartitePureStateAnalysis,
        factorOrder: [CommunicationFactorRole] = [.aliceShare, .bobShare]
    ) throws {
        guard analysis.state == state else {
            throw QuantumMathError.operationNotDefined(
                "Maximally entangled resource analysis must describe the wrapped state."
            )
        }
        guard analysis.leftDimension == analysis.rightDimension else {
            throw QuantumMathError.operationNotDefined(
                "Maximally entangled resources require equal subsystem dimensions."
            )
        }
        guard analysis.classification == .maximallyEntangled else {
            throw QuantumMathError.operationNotDefined(
                "Protocol resource must be maximally entangled."
            )
        }
        guard [2, 3].contains(analysis.leftDimension) else {
            throw QuantumMathError.operationNotDefined(
                "Communication protocols currently support d=2 and d=3 resources."
            )
        }
        guard factorOrder == [.aliceShare, .bobShare] else {
            throw QuantumMathError.operationNotDefined(
                "Communication resources must be ordered as Alice share, then Bob share."
            )
        }

        self.state = state
        self.dimension = analysis.leftDimension
        self.analysis = analysis
        self.factorOrder = factorOrder
    }

    public static func canonical(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault,
        analyzer: EntanglementAnalyzer
    ) throws -> MaximallyEntangledResource {
        let state = try QuantumDomain.maximallyEntangledState(
            dimension: dimension,
            config: config
        )
        return try MaximallyEntangledResource(
            state: state,
            analysis: try analyzer.analyzeBipartitePureState(state)
        )
    }
}

public struct WeylMessage: Hashable, Sendable, Codable, Identifiable {
    public let index: WeylIndex

    public init(dimension: Int, flatIndex: Int) throws {
        self.index = try WeylIndex(dimension: dimension, flatIndex: flatIndex)
    }

    public init(index: WeylIndex) {
        self.index = index
    }

    public var id: Int { flatIndex }
    public var dimension: Int { index.dimension }
    public var shift: Int { index.shift }
    public var phase: Int { index.phase }
    public var flatIndex: Int { index.flatIndex }
}

public struct BellOutcome: Hashable, Sendable, Codable, Identifiable {
    public let index: WeylIndex

    public init(dimension: Int, flatIndex: Int) throws {
        self.index = try WeylIndex(dimension: dimension, flatIndex: flatIndex)
    }

    public init(index: WeylIndex) {
        self.index = index
    }

    public var id: Int { flatIndex }
    public var dimension: Int { index.dimension }
    public var shift: Int { index.shift }
    public var phase: Int { index.phase }
    public var flatIndex: Int { index.flatIndex }
}

public struct ProtocolVerification: Hashable, Sendable, Codable {
    public let succeeded: Bool
    public let metric: Scalar
    public let message: String

    public init(succeeded: Bool, metric: Scalar, message: String) {
        self.succeeded = succeeded
        self.metric = metric
        self.message = message
    }
}

public struct DenseCommunicationTranscript: Hashable, Sendable, Codable {
    public let resource: MaximallyEntangledResource
    public let message: WeylMessage
    public let aliceEncodingOperation: Operator
    public let liftedEncodingOperation: Operator
    public let encodedJointState: Ket
    public let measurementBasis: WeylBellBasis
    public let bobMeasurementObservable: Operator
    public let bobMeasurementState: Ket
    public let bobMeasurementProjector: Operator
    public let decodedMessage: WeylMessage
    public let verification: ProtocolVerification

    public init(
        resource: MaximallyEntangledResource,
        message: WeylMessage,
        aliceEncodingOperation: Operator,
        liftedEncodingOperation: Operator,
        encodedJointState: Ket,
        measurementBasis: WeylBellBasis,
        bobMeasurementObservable: Operator,
        bobMeasurementState: Ket,
        bobMeasurementProjector: Operator,
        decodedMessage: WeylMessage,
        verification: ProtocolVerification
    ) {
        self.resource = resource
        self.message = message
        self.aliceEncodingOperation = aliceEncodingOperation
        self.liftedEncodingOperation = liftedEncodingOperation
        self.encodedJointState = encodedJointState
        self.measurementBasis = measurementBasis
        self.bobMeasurementObservable = bobMeasurementObservable
        self.bobMeasurementState = bobMeasurementState
        self.bobMeasurementProjector = bobMeasurementProjector
        self.decodedMessage = decodedMessage
        self.verification = verification
    }
}

public struct TeleportationTranscript: Hashable, Sendable, Codable {
    public let inputState: Ket
    public let resource: MaximallyEntangledResource
    public let initialState: Ket
    public let outcome: BellOutcome
    public let measurementBasis: WeylBellBasis
    public let measurementState: Ket
    public let measurementProjector: Operator
    public let probability: Scalar
    public let bobStateBeforeCorrection: Ket
    public let correctionOperation: Operator
    public let bobStateAfterCorrection: Ket
    public let fullStateAfterMeasurement: Ket
    public let fullStateAfterCorrection: Ket
    public let verification: ProtocolVerification

    public init(
        inputState: Ket,
        resource: MaximallyEntangledResource,
        initialState: Ket,
        outcome: BellOutcome,
        measurementBasis: WeylBellBasis,
        measurementState: Ket,
        measurementProjector: Operator,
        probability: Scalar,
        bobStateBeforeCorrection: Ket,
        correctionOperation: Operator,
        bobStateAfterCorrection: Ket,
        fullStateAfterMeasurement: Ket,
        fullStateAfterCorrection: Ket,
        verification: ProtocolVerification
    ) {
        self.inputState = inputState
        self.resource = resource
        self.initialState = initialState
        self.outcome = outcome
        self.measurementBasis = measurementBasis
        self.measurementState = measurementState
        self.measurementProjector = measurementProjector
        self.probability = probability
        self.bobStateBeforeCorrection = bobStateBeforeCorrection
        self.correctionOperation = correctionOperation
        self.bobStateAfterCorrection = bobStateAfterCorrection
        self.fullStateAfterMeasurement = fullStateAfterMeasurement
        self.fullStateAfterCorrection = fullStateAfterCorrection
        self.verification = verification
    }
}
