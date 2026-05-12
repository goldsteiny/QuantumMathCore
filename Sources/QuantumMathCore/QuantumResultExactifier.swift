import Foundation
import ExactValueRecoveryCore

public struct QuantumExactificationCacheStats: Hashable, Sendable {
    public let lookupHits: Int
    public let lookupMisses: Int
    public let storedExactValues: Int
    public let storedMisses: Int
}

public final class QuantumExactificationCache: @unchecked Sendable {
    public static let defaultCapacity = 8_192

    private enum Entry {
        case exact(ExactScalarExpression)
        case miss
    }

    private struct Key: Hashable {
        let real: Int64
        let imaginary: Int64
        let profile: String
    }

    private let capacity: Int
    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]
    private var insertionOrder: [Key] = []
    private var lookupHits = 0
    private var lookupMisses = 0

    public init(capacity: Int = QuantumExactificationCache.defaultCapacity) {
        self.capacity = max(1, capacity)
    }

    public func exactify(
        _ scalar: Scalar,
        config: QuantumMathConfig
    ) -> Scalar {
        guard scalar.exactValue == nil else {
            return scalar
        }

        let approximate = scalar.approximateValue
        guard approximate.re.isFinite, approximate.im.isFinite else {
            return scalar
        }

        let key = Self.key(for: approximate, config: config)
        if let cached = cachedEntry(for: key) {
            switch cached {
            case let .exact(expression):
                return .exact(expression)
            case .miss:
                return scalar
            }
        }

        let exactified = ScalarExactificationAdapter.fastExactify(scalar, config: config)
        store(
            exactified.exactValue.map { .exact($0) } ?? .miss,
            for: key
        )
        return exactified
    }

    public func removeAll() {
        lock.lock()
        entries.removeAll()
        insertionOrder.removeAll()
        lookupHits = 0
        lookupMisses = 0
        lock.unlock()
    }

    public var stats: QuantumExactificationCacheStats {
        lock.lock()
        let currentEntries = entries.values
        let exactCount = currentEntries.reduce(0) { count, entry in
            if case .exact = entry {
                return count + 1
            }
            return count
        }
        let missCount = currentEntries.count - exactCount
        let snapshot = QuantumExactificationCacheStats(
            lookupHits: lookupHits,
            lookupMisses: lookupMisses,
            storedExactValues: exactCount,
            storedMisses: missCount
        )
        lock.unlock()
        return snapshot
    }

    private func cachedEntry(for key: Key) -> Entry? {
        lock.lock()
        defer { lock.unlock() }
        guard let entry = entries[key] else {
            lookupMisses += 1
            return nil
        }
        lookupHits += 1
        return entry
    }

    private func store(_ entry: Entry, for key: Key) {
        lock.lock()
        defer { lock.unlock() }

        if entries[key] == nil {
            insertionOrder.append(key)
        }
        entries[key] = entry

        while entries.count > capacity, let oldest = insertionOrder.first {
            insertionOrder.removeFirst()
            entries.removeValue(forKey: oldest)
        }
    }

    private static func key(for value: ComplexNumber, config: QuantumMathConfig) -> Key {
        let tolerance = max(config.exactificationPolicy.candidateDistanceThreshold, .leastNonzeroMagnitude)
        return Key(
            real: quantized(value.re, tolerance: tolerance),
            imaginary: quantized(value.im, tolerance: tolerance),
            profile: exactificationProfile(config)
        )
    }

    private static func quantized(_ value: Double, tolerance: Double) -> Int64 {
        let scaled = (value / tolerance).rounded()
        if scaled >= Double(Int64.max) {
            return Int64.max
        }
        if scaled <= Double(Int64.min) {
            return Int64.min
        }
        return Int64(scaled)
    }

    private static func exactificationProfile(_ config: QuantumMathConfig) -> String {
        [
            "fast-v1",
            "tol=\(config.exactificationPolicy.candidateDistanceThreshold)",
            "families=\(config.exactificationPolicy.enabledFamilies.sorted().joined(separator: ","))",
            "maxCandidates=\(config.exactificationPolicy.maxCandidatesPerVariable)"
        ].joined(separator: ";")
    }
}

public struct QuantumResultExactifier: Sendable {
    public let config: QuantumMathConfig
    public let cache: QuantumExactificationCache

    public init(
        config: QuantumMathConfig,
        cache: QuantumExactificationCache
    ) {
        self.config = config
        self.cache = cache
    }

    public func exactify(_ scalar: Scalar) -> Scalar {
        cache.exactify(scalar, config: config)
    }

    public func exactify(_ ket: Ket) throws -> Ket {
        StateVectorExactificationAdapter
            .exactify(ket, config: config, cache: cache)
            .ket
    }

    public func exactify(_ bra: Bra) throws -> Bra {
        try Bra(
            space: bra.space,
            basis: bra.basis,
            coefficients: bra.coefficients.map { exactify($0) }
        )
    }

    public func exactify(_ operatorValue: Operator) throws -> Operator {
        try Operator(
            domain: operatorValue.domain,
            codomain: operatorValue.codomain,
            columnBasis: operatorValue.columnBasis,
            rowBasis: operatorValue.rowBasis,
            entries: exactify(operatorValue.entries)
        )
    }

    public func exactify(_ matrix: Matrix<Scalar>) -> Matrix<Scalar> {
        Matrix<Scalar>(
            uncheckedRows: matrix.rows,
            cols: matrix.cols,
            values: matrix.values.map { exactify($0) }
        )
    }

    public func exactify(_ value: QuantumValue) throws -> QuantumValue {
        switch value {
        case let .scalar(scalar):
            return .scalar(exactify(scalar))
        case let .ket(ket):
            return .ket(try exactify(ket))
        case let .bra(bra):
            return .bra(try exactify(bra))
        case let .oper(operatorValue):
            return .oper(try exactify(operatorValue))
        }
    }

    public func exactify(_ result: SpectralAnalysisResult) throws -> SpectralAnalysisResult {
        var scalarMetadata: [Int: ExactificationMetadata] = [:]
        var vectorMetadata: [Int: [String: ExactificationMetadata]] = [:]
        let components = try result.decomposition.components.enumerated().map { componentIndex, component in
            let eigenvalue = exactify(component.eigenvalue)
            scalarMetadata[componentIndex] = metadata(for: eigenvalue)

            let eigenvectors = try component.eigenvectors.enumerated().map { vectorIndex, vector in
                let exactified = StateVectorExactificationAdapter.exactify(
                    vector,
                    config: config,
                    cache: cache
                )
                for (coefficientIndex, coefficientMetadata) in exactified.metadataByCoefficient {
                    vectorMetadata[componentIndex, default: [:]]["v\(vectorIndex).c\(coefficientIndex)"] = coefficientMetadata
                }
                return exactified.ket
            }

            return EigenComponent(
                eigenvalue: eigenvalue,
                multiplicity: component.multiplicity,
                eigenvectors: eigenvectors
            )
        }

        return SpectralAnalysisResult(
            decomposition: SpectralDecomposition(
                space: result.decomposition.space,
                basis: result.decomposition.basis,
                components: components
            ),
            exactificationMetadataByComponent: scalarMetadata,
            vectorExactificationMetadataByComponent: vectorMetadata
        )
    }

    public func exactify(_ result: SingularValueAnalysisResult) throws -> SingularValueAnalysisResult {
        var scalarMetadata: [Int: ExactificationMetadata] = [:]
        var leftVectorMetadata: [Int: [Int: ExactificationMetadata]] = [:]
        var rightVectorMetadata: [Int: [Int: ExactificationMetadata]] = [:]
        let components = try result.decomposition.components.enumerated().map { index, component in
            let singularValue = exactify(component.singularValue)
            scalarMetadata[index] = metadata(for: singularValue)

            let left = StateVectorExactificationAdapter.exactify(
                component.leftVector,
                config: config,
                cache: cache
            )
            let right = StateVectorExactificationAdapter.exactify(
                component.rightVector,
                config: config,
                cache: cache
            )
            leftVectorMetadata[index] = left.metadataByCoefficient
            rightVectorMetadata[index] = right.metadataByCoefficient

            return SingularValueComponent(
                singularValue: singularValue,
                leftVector: left.ket,
                rightVector: right.ket
            )
        }

        return SingularValueAnalysisResult(
            decomposition: SingularValueDecomposition(
                domain: result.decomposition.domain,
                codomain: result.decomposition.codomain,
                columnBasis: result.decomposition.columnBasis,
                rowBasis: result.decomposition.rowBasis,
                components: components
            ),
            exactificationMetadataByComponent: scalarMetadata,
            leftVectorExactificationMetadataByComponent: leftVectorMetadata,
            rightVectorExactificationMetadataByComponent: rightVectorMetadata
        )
    }

    public func exactify(_ result: MeasurementAnalysisResult) throws -> MeasurementAnalysisResult {
        try MeasurementAnalysisResult(
            observable: ProjectiveMeasurement(
                observable: try exactify(result.observable.observable),
                config: config
            ),
            state: try exactify(result.state),
            outcomes: try result.outcomes.map { outcome in
                MeasurementOutcome(
                    eigenvalue: exactify(outcome.eigenvalue),
                    multiplicity: outcome.multiplicity,
                    projector: try exactify(outcome.projector),
                    probability: exactify(outcome.probability),
                    postMeasurementState: try outcome.postMeasurementState.map { try exactify($0) },
                    isZeroProbability: outcome.isZeroProbability
                )
            },
            probabilityTotal: exactify(result.probabilityTotal),
            moments: MeasurementMoments(
                expectedValue: exactify(result.moments.expectedValue),
                expectedSquaredValue: exactify(result.moments.expectedSquaredValue),
                variance: exactify(result.moments.variance),
                standardDeviation: exactify(result.moments.standardDeviation)
            ),
            posterior: MeasurementPosterior(
                state: try exactify(result.posterior.state),
                purity: exactify(result.posterior.purity)
            )
        )
    }

    public func exactify(_ state: MeasurementState) throws -> MeasurementState {
        switch state {
        case let .pure(ket):
            return .pure(try exactify(ket))
        case let .density(operatorValue):
            return .density(try exactify(operatorValue))
        }
    }

    private func metadata(for scalar: Scalar) -> ExactificationMetadata {
        ExactificationMetadata(
            status: scalar.isApproximate ? .approximateRetained : .verifiedExact,
            witness: nil
        )
    }
}
