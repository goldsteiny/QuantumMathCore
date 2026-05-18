import Foundation

public enum EntanglementClassification: String, Hashable, Sendable, Codable {
    case product
    case entangled
    case maximallyEntangled
}

public struct SchmidtFactorPartition: Hashable, Sendable, Codable {
    public let leftFactors: TensorFactorSet

    public init(leftFactors: TensorFactorSet) {
        self.leftFactors = leftFactors
    }

    public init(leftRawOffsets: [Int]) throws {
        self.init(leftFactors: try TensorFactorSet(rawOffsets: leftRawOffsets))
    }

    public static func firstFactorVsRest(for state: Ket) throws -> SchmidtFactorPartition {
        guard state.space.factors.count >= 2 else {
            throw QuantumMathError.operationNotDefined(
                "Schmidt decomposition requires a pure ket on at least two tensor factors."
            )
        }
        return try SchmidtFactorPartition(leftRawOffsets: [0])
    }
}

public struct ResolvedSchmidtFactorPartition: Hashable, Sendable, Codable {
    public let leftOffsets: [TensorFactorOffset]
    public let rightOffsets: [TensorFactorOffset]
    public let leftSpace: Space
    public let rightSpace: Space
    public let leftBasis: Basis
    public let rightBasis: Basis
    public let leftDimension: Int
    public let rightDimension: Int
}

public struct SchmidtComponent: Hashable, Sendable, Codable {
    public let coefficient: Scalar
    public let leftVector: Ket
    public let rightVector: Ket

    public init(
        coefficient: Scalar,
        leftVector: Ket,
        rightVector: Ket
    ) {
        self.coefficient = coefficient
        self.leftVector = leftVector
        self.rightVector = rightVector
    }
}

public struct ReducedSpectrumRelation: Hashable, Sendable, Codable {
    public let squaredSchmidtCoefficients: [Scalar]
    public let leftEigenvalues: [Scalar]
    public let rightEigenvalues: [Scalar]
    public let maxAbsoluteDeviation: Double
    public let matchesWithinTolerance: Bool

    public init(
        squaredSchmidtCoefficients: [Scalar],
        leftEigenvalues: [Scalar],
        rightEigenvalues: [Scalar],
        maxAbsoluteDeviation: Double,
        matchesWithinTolerance: Bool
    ) {
        self.squaredSchmidtCoefficients = squaredSchmidtCoefficients
        self.leftEigenvalues = leftEigenvalues
        self.rightEigenvalues = rightEigenvalues
        self.maxAbsoluteDeviation = maxAbsoluteDeviation
        self.matchesWithinTolerance = matchesWithinTolerance
    }
}

public struct SchmidtDecompositionAnalysis: Hashable, Sendable, Codable {
    public let state: Ket
    public let partition: ResolvedSchmidtFactorPartition
    public let schmidtRank: Int
    public let schmidtCoefficients: [Scalar]
    public let components: [SchmidtComponent]
    public let classification: EntanglementClassification
    public let reconstructionResidual: Double
    public let reducedSpectrumRelation: ReducedSpectrumRelation

    public init(
        state: Ket,
        partition: ResolvedSchmidtFactorPartition,
        schmidtRank: Int,
        schmidtCoefficients: [Scalar],
        components: [SchmidtComponent],
        classification: EntanglementClassification,
        reconstructionResidual: Double,
        reducedSpectrumRelation: ReducedSpectrumRelation
    ) {
        self.state = state
        self.partition = partition
        self.schmidtRank = schmidtRank
        self.schmidtCoefficients = schmidtCoefficients
        self.components = components
        self.classification = classification
        self.reconstructionResidual = reconstructionResidual
        self.reducedSpectrumRelation = reducedSpectrumRelation
    }
}

public struct SchmidtSummaryAnalysis: Hashable, Sendable, Codable {
    public let state: Ket
    public let partition: ResolvedSchmidtFactorPartition
    public let schmidtRank: Int
    public let schmidtCoefficients: [Scalar]
    public let classification: EntanglementClassification

    public init(
        state: Ket,
        partition: ResolvedSchmidtFactorPartition,
        schmidtRank: Int,
        schmidtCoefficients: [Scalar],
        classification: EntanglementClassification
    ) {
        self.state = state
        self.partition = partition
        self.schmidtRank = schmidtRank
        self.schmidtCoefficients = schmidtCoefficients
        self.classification = classification
    }
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

private struct PreparedSchmidtDecomposition {
    let partition: ResolvedSchmidtFactorPartition
    let coefficientMatrix: Matrix<Scalar>
    let raw: SingularValueDecompositionRaw
}

public struct EntanglementAnalyzer: Sendable {
    public let config: QuantumMathConfig
    public let backend: any SingularValueSolver

    public init(config: QuantumMathConfig, backend: any SingularValueSolver) {
        self.config = config
        self.backend = backend
    }

    public func analyzeSchmidtDecomposition(_ state: Ket) throws -> SchmidtDecompositionAnalysis {
        try analyzeSchmidtDecomposition(
            state,
            partition: SchmidtFactorPartition.firstFactorVsRest(for: state)
        )
    }

    public func analyzeSchmidtSummary(_ state: Ket) throws -> SchmidtSummaryAnalysis {
        try analyzeSchmidtSummary(
            state,
            partition: SchmidtFactorPartition.firstFactorVsRest(for: state)
        )
    }

    public func analyzeSchmidtSummary(
        _ state: Ket,
        partition: SchmidtFactorPartition
    ) throws -> SchmidtSummaryAnalysis {
        let prepared = try prepareSchmidtDecomposition(state, partition: partition)
        let schmidtCoefficients = significantSchmidtCoefficients(from: prepared.raw)
        let classification = classifySchmidtState(
            partition: prepared.partition,
            schmidtCoefficients: schmidtCoefficients
        )

        return SchmidtSummaryAnalysis(
            state: state,
            partition: prepared.partition,
            schmidtRank: schmidtCoefficients.count,
            schmidtCoefficients: schmidtCoefficients,
            classification: classification
        )
    }

    public func analyzeSchmidtDecomposition(
        _ state: Ket,
        partition: SchmidtFactorPartition
    ) throws -> SchmidtDecompositionAnalysis {
        let prepared = try prepareSchmidtDecomposition(state, partition: partition)

        var components = try schmidtComponents(
            raw: prepared.raw,
            partition: prepared.partition
        )
        components.sort {
            $0.coefficient.approximateValue.re > $1.coefficient.approximateValue.re
        }
        let schmidtCoefficients = components.map(\.coefficient)
        let schmidtRank = schmidtCoefficients.count
        let classification = classifySchmidtState(
            partition: prepared.partition,
            schmidtCoefficients: schmidtCoefficients
        )

        let reconstructionResidual = try schmidtReconstructionResidual(
            source: prepared.coefficientMatrix,
            components: components,
            leftDimension: prepared.partition.leftDimension,
            rightDimension: prepared.partition.rightDimension
        )
        let reducedSpectrumRelation = try reducedSpectrumRelation(
            state: state,
            partition: prepared.partition,
            coefficients: schmidtCoefficients
        )

        return SchmidtDecompositionAnalysis(
            state: state,
            partition: prepared.partition,
            schmidtRank: schmidtRank,
            schmidtCoefficients: schmidtCoefficients,
            components: components,
            classification: classification,
            reconstructionResidual: reconstructionResidual,
            reducedSpectrumRelation: reducedSpectrumRelation
        )
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

        let summary = try analyzeSchmidtSummary(
            state,
            partition: SchmidtFactorPartition(leftFactors: try TensorFactorSet(rawOffsets: [0]))
        )

        return BipartitePureStateAnalysis(
            state: state,
            leftDimension: summary.partition.leftDimension,
            rightDimension: summary.partition.rightDimension,
            schmidtRank: summary.schmidtRank,
            schmidtCoefficients: summary.schmidtCoefficients,
            classification: summary.classification
        )
    }

    private func prepareSchmidtDecomposition(
        _ state: Ket,
        partition: SchmidtFactorPartition
    ) throws -> PreparedSchmidtDecomposition {
        guard state.space.factors.count >= 2 else {
            throw QuantumMathError.operationNotDefined(
                "Schmidt decomposition requires a pure ket on at least two tensor factors."
            )
        }
        guard state.basis.isComputationalCoordinateBasis else {
            throw QuantumMathError.operationNotDefined(
                "Schmidt decomposition requires computational-coordinate basis."
            )
        }
        guard abs(QuantumDomain.normSquared(state) - 1) <= config.scalarComparisonEpsilon else {
            throw QuantumMathError.operationNotDefined("Schmidt decomposition requires a normalized pure ket.")
        }

        let resolvedPartition = try resolvedPartition(for: state, partition: partition)
        let matrix = try coefficientMatrix(for: state, partition: resolvedPartition)
        let raw = try backend.decompose(matrix)
        return PreparedSchmidtDecomposition(
            partition: resolvedPartition,
            coefficientMatrix: matrix,
            raw: raw
        )
    }

    private func significantSchmidtCoefficients(from raw: SingularValueDecompositionRaw) -> [Scalar] {
        raw.singularValues.prefix(raw.rank)
            .map(schmidtCoefficient(from:))
            .sorted { lhs, rhs in
                lhs.approximateValue.re > rhs.approximateValue.re
            }
            .filter { coefficient in
                coefficient.approximateValue.re > config.scalarComparisonEpsilon
            }
    }

    private func classifySchmidtState(
        partition: ResolvedSchmidtFactorPartition,
        schmidtCoefficients: [Scalar]
    ) -> EntanglementClassification {
        let schmidtRank = schmidtCoefficients.count
        guard schmidtRank > 1 else {
            return .product
        }

        if partition.leftDimension == partition.rightDimension,
           schmidtRank == partition.leftDimension,
           schmidtCoefficients.allSatisfy({
               abs($0.approximateValue.re - (1 / Foundation.sqrt(Double(partition.leftDimension))))
                   <= config.scalarComparisonEpsilon
           }) {
            return .maximallyEntangled
        }

        return .entangled
    }

    private func schmidtComponents(
        raw: SingularValueDecompositionRaw,
        partition: ResolvedSchmidtFactorPartition
    ) throws -> [SchmidtComponent] {
        try (0..<raw.rank).compactMap { index -> SchmidtComponent? in
            let coefficient = schmidtCoefficient(from: raw.singularValues[index])
            guard coefficient.approximateValue.re > config.scalarComparisonEpsilon else {
                return nil
            }

            var left = raw.leftVector(at: index)
            var rightSingular = raw.rightVector(at: index)
            schmidtPhaseCanonicalize(left: &left, right: &rightSingular)

            let leftVector = try Ket(
                space: partition.leftSpace,
                basis: partition.leftBasis,
                coefficients: left.map { .approx(schmidtSanitized($0)) }
            )
            let rightVector = try Ket(
                space: partition.rightSpace,
                basis: partition.rightBasis,
                coefficients: rightSingular.map { value in
                    .approx(schmidtSanitized(value.conjugated))
                }
            )

            return SchmidtComponent(
                coefficient: coefficient,
                leftVector: leftVector,
                rightVector: rightVector
            )
        }
    }

    private func schmidtCoefficient(from singularValue: Double) -> Scalar {
        Scalar.approx(
            ComplexNumber(
                re: schmidtSanitizedNonNegative(singularValue),
                im: 0
            )
        )
    }

    private func resolvedPartition(
        for state: Ket,
        partition: SchmidtFactorPartition
    ) throws -> ResolvedSchmidtFactorPartition {
        let factorCount = state.space.factors.count
        let leftOffsets = partition.leftFactors.offsets

        guard leftOffsets.allSatisfy({ $0.rawValue < factorCount }) else {
            throw QuantumMathError.operationNotDefined(
                "Selected Schmidt partition factor is out of range for the state space."
            )
        }
        guard leftOffsets.count < factorCount else {
            throw QuantumMathError.operationNotDefined(
                "Schmidt partition must leave at least one factor on each side."
            )
        }

        let rightOffsets = try (0..<factorCount)
            .map { try TensorFactorOffset(rawValue: $0) }
            .filter { !partition.leftFactors.contains($0) }

        let leftFactors = leftOffsets.map { state.space.factors[$0.rawValue] }
        let rightFactors = rightOffsets.map { state.space.factors[$0.rawValue] }
        let leftKinds = leftOffsets.map { state.basis.factorKinds[$0.rawValue] }
        let rightKinds = rightOffsets.map { state.basis.factorKinds[$0.rawValue] }

        let leftSpace = try Space(
            validating: leftFactors,
            maxComputableDimension: config.maxComputableDimension
        )
        let rightSpace = try Space(
            validating: rightFactors,
            maxComputableDimension: config.maxComputableDimension
        )

        return try ResolvedSchmidtFactorPartition(
            leftOffsets: leftOffsets,
            rightOffsets: rightOffsets,
            leftSpace: leftSpace,
            rightSpace: rightSpace,
            leftBasis: Basis(space: leftSpace, factorKinds: leftKinds),
            rightBasis: Basis(space: rightSpace, factorKinds: rightKinds),
            leftDimension: leftSpace.dimension,
            rightDimension: rightSpace.dimension
        )
    }

    private func coefficientMatrix(
        for state: Ket,
        partition: ResolvedSchmidtFactorPartition
    ) throws -> Matrix<Scalar> {
        let sourceFactorDimensions = state.space.factors.map(\.dimension)
        let leftDimensions = partition.leftOffsets.map { sourceFactorDimensions[$0.rawValue] }
        let rightDimensions = partition.rightOffsets.map { sourceFactorDimensions[$0.rawValue] }

        var values = Array(
            repeating: Scalar.zero,
            count: partition.leftDimension * partition.rightDimension
        )

        for sourceFlatIndex in state.coefficients.indices {
            let fullIndices = try TensorIndexing.unflatten(
                index: sourceFlatIndex,
                factorDimensions: sourceFactorDimensions
            )
            let leftIndices = partition.leftOffsets.map { fullIndices[$0.rawValue] }
            let rightIndices = partition.rightOffsets.map { fullIndices[$0.rawValue] }
            let leftFlat = try TensorIndexing.flatten(indices: leftIndices, factorDimensions: leftDimensions)
            let rightFlat = try TensorIndexing.flatten(indices: rightIndices, factorDimensions: rightDimensions)
            values[(leftFlat * partition.rightDimension) + rightFlat] = state.coefficients[sourceFlatIndex]
        }

        return try Matrix(
            rows: partition.leftDimension,
            cols: partition.rightDimension,
            values: values
        )
    }

    private func schmidtReconstructionResidual(
        source: Matrix<Scalar>,
        components: [SchmidtComponent],
        leftDimension: Int,
        rightDimension: Int
    ) throws -> Double {
        var reconstructedValues = Array(repeating: Scalar.zero, count: leftDimension * rightDimension)

        for row in 0..<leftDimension {
            for col in 0..<rightDimension {
                let reconstructedEntry = components.reduce(Scalar.zero) { partial, component in
                    partial
                        + (component.coefficient
                            * component.leftVector.coefficients[row]
                            * component.rightVector.coefficients[col])
                }
                reconstructedValues[(row * rightDimension) + col] = reconstructedEntry
            }
        }

        let reconstructed = try Matrix(
            rows: leftDimension,
            cols: rightDimension,
            values: reconstructedValues
        )
        let residualSquared = zip(source.values, reconstructed.values).reduce(0.0) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + delta.magnitudeSquaredApproximate
        }
        return Foundation.sqrt(residualSquared)
    }

    private func reducedSpectrumRelation(
        state: Ket,
        partition: ResolvedSchmidtFactorPartition,
        coefficients: [Scalar]
    ) throws -> ReducedSpectrumRelation {
        let density = try QuantumDomain.pureDensityState(from: .ket(state), config: config)
        let leftReduced = try QuantumDomain.partialTrace(
            density,
            tracing: PartialTraceSelection(tracedRawOffsets: partition.rightOffsets.map(\.rawValue))
        )
        let rightReduced = try QuantumDomain.partialTrace(
            density,
            tracing: PartialTraceSelection(tracedRawOffsets: partition.leftOffsets.map(\.rawValue))
        )

        let leftEigenvalues = try backend.decompose(leftReduced.entries).singularValues.map {
            Scalar.approx(ComplexNumber(re: schmidtSanitizedNonNegative($0), im: 0))
        }
        let rightEigenvalues = try backend.decompose(rightReduced.entries).singularValues.map {
            Scalar.approx(ComplexNumber(re: schmidtSanitizedNonNegative($0), im: 0))
        }
        let squaredCoefficients = coefficients.map { coefficient in
            let value = coefficient.approximateValue.re
            return Scalar.approx(ComplexNumber(re: max(0, value * value), im: 0))
        }

        let expectedLeft = paddedSpectrum(
            from: squaredCoefficients,
            count: leftEigenvalues.count
        )
        let expectedRight = paddedSpectrum(
            from: squaredCoefficients,
            count: rightEigenvalues.count
        )
        let leftDeviation = maxSpectrumDeviation(lhs: expectedLeft, rhs: leftEigenvalues)
        let rightDeviation = maxSpectrumDeviation(lhs: expectedRight, rhs: rightEigenvalues)
        let maxDeviation = max(leftDeviation, rightDeviation)

        return ReducedSpectrumRelation(
            squaredSchmidtCoefficients: squaredCoefficients,
            leftEigenvalues: leftEigenvalues,
            rightEigenvalues: rightEigenvalues,
            maxAbsoluteDeviation: maxDeviation,
            matchesWithinTolerance: maxDeviation <= (config.scalarComparisonEpsilon * 8)
        )
    }

    private func paddedSpectrum(
        from squaredCoefficients: [Scalar],
        count: Int
    ) -> [Scalar] {
        let sorted = squaredCoefficients.sorted {
            $0.approximateValue.re > $1.approximateValue.re
        }
        if sorted.count >= count {
            return Array(sorted.prefix(count))
        }
        return sorted + Array(repeating: .zero, count: count - sorted.count)
    }

    private func maxSpectrumDeviation(lhs: [Scalar], rhs: [Scalar]) -> Double {
        let sortedLHS = lhs.sorted {
            $0.approximateValue.re > $1.approximateValue.re
        }
        let sortedRHS = rhs.sorted {
            $0.approximateValue.re > $1.approximateValue.re
        }
        return zip(sortedLHS, sortedRHS).reduce(0.0) { partial, pair in
            max(partial, abs(pair.0.approximateValue.re - pair.1.approximateValue.re))
        }
    }
}

private func schmidtPhaseCanonicalize(left: inout [ComplexNumber], right: inout [ComplexNumber]) {
    let pivot = schmidtCanonicalPhasePivot(in: right) ?? schmidtCanonicalPhasePivot(in: left)
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
    left = left.map { schmidtSanitized($0 * inversePhase) }
    right = right.map { schmidtSanitized($0 * inversePhase) }
}

private func schmidtCanonicalPhasePivot(in vector: [ComplexNumber]) -> ComplexNumber? {
    guard let pivotIndex = vector.indices.max(by: { lhs, rhs in
        vector[lhs].magnitudeSquared < vector[rhs].magnitudeSquared
    }) else {
        return nil
    }
    return vector[pivotIndex]
}

private func schmidtSanitized(_ value: ComplexNumber) -> ComplexNumber {
    ComplexNumber(
        re: value.re == 0 ? 0 : value.re,
        im: value.im == 0 ? 0 : value.im
    )
}

private func schmidtSanitizedNonNegative(_ value: Double) -> Double {
    if !value.isFinite {
        return value
    }
    if abs(value) <= 1e-12 {
        return 0
    }
    return max(0, value)
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
