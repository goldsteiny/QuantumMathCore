import Foundation

public struct TensorFactorOffset: Hashable, Comparable, Sendable, Codable {
    public let rawValue: Int

    public init(rawValue: Int) throws {
        guard rawValue >= 0 else {
            throw QuantumMathError.operationNotDefined("Tensor factor offsets must be non-negative.")
        }
        self.rawValue = rawValue
    }

    public static func < (lhs: TensorFactorOffset, rhs: TensorFactorOffset) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public struct TensorFactorSet: Hashable, Sendable, Codable {
    public let offsets: [TensorFactorOffset]

    public init(offsets: [TensorFactorOffset]) throws {
        let sorted = offsets.sorted()
        guard !sorted.isEmpty else {
            throw QuantumMathError.operationNotDefined("Choose at least one tensor factor.")
        }
        guard Set(sorted.map(\.rawValue)).count == sorted.count else {
            throw QuantumMathError.operationNotDefined("Tensor factor selections cannot contain duplicates.")
        }
        self.offsets = sorted
    }

    public init(rawOffsets: [Int]) throws {
        try self.init(offsets: rawOffsets.map { try TensorFactorOffset(rawValue: $0) })
    }

    public func contains(_ offset: TensorFactorOffset) -> Bool {
        offsets.contains(offset)
    }
}

public struct PartialTraceSelection: Hashable, Sendable, Codable {
    public let tracedFactors: TensorFactorSet

    public init(tracedFactors: TensorFactorSet) {
        self.tracedFactors = tracedFactors
    }

    public init(tracedRawOffsets: [Int]) throws {
        self.init(tracedFactors: try TensorFactorSet(rawOffsets: tracedRawOffsets))
    }
}

public struct TensorProductEndomorphism: Hashable, Sendable {
    public let operatorValue: Operator
    public let space: Space
    public let basis: Basis
    public let factorDimensions: [Int]

    public init(_ operatorValue: Operator) throws {
        guard operatorValue.domain == operatorValue.codomain else {
            throw QuantumMathError.operationNotDefined(
                "Partial trace requires a square operator on one tensor-product space."
            )
        }
        guard operatorValue.rowBasis == operatorValue.columnBasis else {
            throw QuantumMathError.operationNotDefined(
                "Partial trace requires matching row and column bases."
            )
        }
        guard case .tensor = operatorValue.domain else {
            throw QuantumMathError.operationNotDefined("Partial trace requires a tensor-product space.")
        }
        self.operatorValue = operatorValue
        self.space = operatorValue.domain
        self.basis = operatorValue.rowBasis
        self.factorDimensions = operatorValue.domain.factors.map(\.dimension)
    }
}

public struct ResolvedPartialTraceSelection: Hashable, Sendable {
    public let tracedOffsets: [TensorFactorOffset]
    public let retainedOffsets: [TensorFactorOffset]
    public let retainedSpace: Space
    public let retainedBasis: Basis

    public init(selection: PartialTraceSelection, source: TensorProductEndomorphism) throws {
        let factorCount = source.factorDimensions.count
        let tracedOffsets = selection.tracedFactors.offsets
        guard tracedOffsets.allSatisfy({ $0.rawValue < factorCount }) else {
            throw QuantumMathError.operationNotDefined(
                "Selected tensor factor is out of range for the source space."
            )
        }
        guard tracedOffsets.count < factorCount else {
            throw QuantumMathError.operationNotDefined(
                "Partial trace must retain at least one tensor factor."
            )
        }

        let retainedOffsets = try (0..<factorCount)
            .map { try TensorFactorOffset(rawValue: $0) }
            .filter { !selection.tracedFactors.contains($0) }
        let retainedFactors = retainedOffsets.map { source.space.factors[$0.rawValue] }
        let retainedKinds = retainedOffsets.map { source.basis.factorKinds[$0.rawValue] }
        let retainedSpace = try Space(
            validating: retainedFactors,
            maxComputableDimension: source.space.dimension
        )

        self.tracedOffsets = tracedOffsets
        self.retainedOffsets = retainedOffsets
        self.retainedSpace = retainedSpace
        self.retainedBasis = try Basis(space: retainedSpace, factorKinds: retainedKinds)
    }
}

public enum TensorIndexing {
    public static func flatten(indices: [Int], factorDimensions: [Int]) throws -> Int {
        guard indices.count == factorDimensions.count else {
            throw QuantumMathError.operationNotDefined("Tensor index rank mismatch.")
        }
        var flat = 0
        var stride = 1
        for (index, dimension) in zip(indices.reversed(), factorDimensions.reversed()) {
            guard index >= 0, index < dimension else {
                throw QuantumMathError.operationNotDefined("Tensor index is out of range.")
            }
            flat += index * stride
            stride *= dimension
        }
        return flat
    }

    public static func unflatten(index: Int, factorDimensions: [Int]) throws -> [Int] {
        let totalDimension = factorDimensions.reduce(1, *)
        guard index >= 0, index < totalDimension else {
            throw QuantumMathError.operationNotDefined("Flat tensor index is out of range.")
        }
        var remaining = index
        var result = Array(repeating: 0, count: factorDimensions.count)
        for offset in factorDimensions.indices.reversed() {
            let dimension = factorDimensions[offset]
            result[offset] = remaining % dimension
            remaining /= dimension
        }
        return result
    }
}

