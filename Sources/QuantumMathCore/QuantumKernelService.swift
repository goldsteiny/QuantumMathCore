import Foundation

public enum QuantumAnalysisRequestKind: String, Hashable, Sendable, Codable {
    case value
    case matrixPreview
    case traits
    case spectrum
    case svd
    case entanglement
    case measurement
    case exportRender
}

public struct QuantumAnalysisIdentity: Hashable, Sendable, Codable {
    public let inputID: String
    public let configID: String
    public let backendID: String
    public let requestKind: QuantumAnalysisRequestKind

    public init(
        inputID: String,
        configID: String,
        backendID: String,
        requestKind: QuantumAnalysisRequestKind
    ) {
        self.inputID = inputID
        self.configID = configID
        self.backendID = backendID
        self.requestKind = requestKind
    }
}

public enum QuantumAnalysisFailureKind: Hashable, Sendable, Codable {
    case invalidInput
    case unsupportedOperation
    case backendFailure(BackendFailure)
    case exactificationUnresolved
}

public struct QuantumAnalysisFailure: Error, Hashable, Sendable, Codable {
    public let identity: QuantumAnalysisIdentity?
    public let kind: QuantumAnalysisFailureKind
    public let message: String

    public init(
        identity: QuantumAnalysisIdentity?,
        kind: QuantumAnalysisFailureKind,
        message: String
    ) {
        self.identity = identity
        self.kind = kind
        self.message = message
    }
}

public enum QuantumAnalysisResult: Hashable, Sendable, Codable {
    case value(QuantumValue)
    case matrixPreview(Matrix<Scalar>)
    case traits(Set<OperatorTrait>)
    case spectrum(SpectralAnalysisResult)
    case svd(SingularValueAnalysisResult)
    case entanglement(BipartitePureStateAnalysis)
    case measurement(MeasurementAnalysisResult)
    case exportRender(String)
}

public struct QuantumAnalysisEnvelope: Hashable, Sendable, Codable {
    public let identity: QuantumAnalysisIdentity
    public let result: QuantumAnalysisResult

    public init(
        identity: QuantumAnalysisIdentity,
        result: QuantumAnalysisResult
    ) {
        self.identity = identity
        self.result = result
    }
}

public struct QuantumKernelService: Sendable {
    public let config: QuantumMathConfig
    public let backends: QuantumMathBackends
    public let backendID: String
    public let exactificationCache: QuantumExactificationCache

    public init(
        config: QuantumMathConfig,
        backends: QuantumMathBackends,
        backendID: String = "quantum.backends.default",
        exactificationCache: QuantumExactificationCache = QuantumExactificationCache()
    ) {
        self.config = config
        self.backends = backends
        self.backendID = backendID
        self.exactificationCache = exactificationCache
    }

    public var configID: String {
        [
            "dim=\(config.maxComputableDimension)",
            "matrix=\(config.matrixTraitEpsilon)",
            "scalar=\(config.scalarComparisonEpsilon)",
            "spectral=\(config.spectralResidualThreshold)",
            "svd=\(config.svdResidualThreshold)",
            "exact=\(config.exactificationPolicy.candidateDistanceThreshold)"
        ].joined(separator: ";")
    }

    public var operations: QuantumOperation {
        QuantumOperation(config: config)
    }

    public var spectralAnalyzer: SpectralAnalyzer {
        SpectralAnalyzer(config: config, backend: backends.hermitianEigenSolver)
    }

    public var singularValueAnalyzer: SingularValueAnalyzer {
        SingularValueAnalyzer(config: config, backend: backends.singularValueSolver)
    }

    public var entanglementAnalyzer: EntanglementAnalyzer {
        EntanglementAnalyzer(config: config, backend: backends.singularValueSolver)
    }

    public var measurementAnalyzer: MeasurementAnalyzer {
        MeasurementAnalyzer(config: config, spectralAnalyzer: spectralAnalyzer)
    }

    public var resultExactifier: QuantumResultExactifier {
        QuantumResultExactifier(config: config, cache: exactificationCache)
    }

    public func evaluate(
        expression: QuantumExpression,
        resolver: ValueResolver
    ) throws -> QuantumValue {
        try operations.evaluate(expression: expression, resolver: resolver)
    }

    public func evaluate(
        inputID: String,
        expression: QuantumExpression,
        resolver: ValueResolver
    ) throws -> QuantumAnalysisEnvelope {
        QuantumAnalysisEnvelope(
            identity: analysisIdentity(inputID: inputID, requestKind: .value),
            result: .value(try evaluate(expression: expression, resolver: resolver))
        )
    }

    public func analyzeSpectrum(_ operatorValue: Operator) throws -> SpectralAnalysisResult {
        try spectralAnalyzer.hermitianDecomposition(operatorValue)
    }

    public func analyzeSpectrum(
        inputID: String,
        _ operatorValue: Operator
    ) throws -> QuantumAnalysisEnvelope {
        QuantumAnalysisEnvelope(
            identity: analysisIdentity(inputID: inputID, requestKind: .spectrum),
            result: .spectrum(try analyzeSpectrum(operatorValue))
        )
    }

    public func analyzeSVD(_ operatorValue: Operator) throws -> SingularValueAnalysisResult {
        try singularValueAnalyzer.decompose(operatorValue)
    }

    public func analyzeSVD(
        inputID: String,
        _ operatorValue: Operator
    ) throws -> QuantumAnalysisEnvelope {
        QuantumAnalysisEnvelope(
            identity: analysisIdentity(inputID: inputID, requestKind: .svd),
            result: .svd(try analyzeSVD(operatorValue))
        )
    }

    public func analyzeEntanglement(_ state: Ket) throws -> BipartitePureStateAnalysis {
        try entanglementAnalyzer.analyzeBipartitePureState(state)
    }

    public func analyzeEntanglement(
        inputID: String,
        _ state: Ket
    ) throws -> QuantumAnalysisEnvelope {
        QuantumAnalysisEnvelope(
            identity: analysisIdentity(inputID: inputID, requestKind: .entanglement),
            result: .entanglement(try analyzeEntanglement(state))
        )
    }

    public func analyzeMeasurement(
        observable: Operator,
        state: MeasurementState
    ) throws -> MeasurementAnalysisResult {
        try measurementAnalyzer.analyze(observable: observable, state: state)
    }

    public func analyzeMeasurement(
        inputID: String,
        observable: Operator,
        state: MeasurementState
    ) throws -> QuantumAnalysisEnvelope {
        QuantumAnalysisEnvelope(
            identity: analysisIdentity(inputID: inputID, requestKind: .measurement),
            result: .measurement(try analyzeMeasurement(observable: observable, state: state))
        )
    }

    public func analysisIdentity(
        inputID: String,
        requestKind: QuantumAnalysisRequestKind
    ) -> QuantumAnalysisIdentity {
        QuantumAnalysisIdentity(
            inputID: inputID,
            configID: configID,
            backendID: backendID,
            requestKind: requestKind
        )
    }
}
