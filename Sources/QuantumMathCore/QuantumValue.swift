import Foundation

public struct Bra: Hashable, Sendable, Codable {
    public let space: Space
    public let basis: Basis
    public let coefficients: [Scalar]

    public init(space: Space, basis: Basis, coefficients: [Scalar]) throws {
        guard basis.space == space else {
            throw QuantumMathError.incompatibleBases(lhs: basis, rhs: .computational(for: space))
        }
        guard coefficients.count == space.dimension else {
            throw QuantumMathError.invalidCoefficientCount(
                expected: space.dimension,
                actual: coefficients.count
            )
        }
        self.space = space
        self.basis = basis
        self.coefficients = coefficients
    }
}

public enum QuantumValue: Hashable, Sendable, Codable {
    case scalar(Scalar)
    case ket(Ket)
    case bra(Bra)
    case oper(Operator)

    public var containsApproximation: Bool {
        switch self {
        case let .scalar(scalar):
            return scalar.isApproximate
        case let .ket(ket):
            return ket.coefficients.contains(where: \.isApproximate)
        case let .bra(bra):
            return bra.coefficients.contains(where: \.isApproximate)
        case let .oper(operatorValue):
            return operatorValue.entries.values.contains(where: \.isApproximate)
        }
    }
}

public enum WeightedSumNormalization: String, Hashable, Sendable, Codable, CaseIterable {
    case none
    case stateUnitInnerProduct
    case operatorUnitTrace
}

public enum OperatorTrait: String, Hashable, Sendable, Codable, CaseIterable {
    case hermitian
    case unitary
    case projector
    case diagonal
    case identity
}

public enum DensityPurity: Hashable, Sendable, Codable {
    case pure
    case mixed
}

public struct QubitBlochVector: Hashable, Sendable, Codable {
    public let x: Scalar
    public let y: Scalar
    public let z: Scalar

    public init(x: Scalar, y: Scalar, z: Scalar) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public struct QubitBlochAngles: Hashable, Sendable, Codable {
    public let thetaRadians: Scalar
    public let phiRadians: Scalar

    public init(thetaRadians: Scalar, phiRadians: Scalar) {
        self.thetaRadians = thetaRadians
        self.phiRadians = phiRadians
    }
}

public struct DensityOperatorSummary: Hashable, Sendable, Codable {
    public let purity: DensityPurity
    public let recoveredKet: Ket?
    public let qubitBlochVector: QubitBlochVector?

    public init(
        purity: DensityPurity,
        recoveredKet: Ket?,
        qubitBlochVector: QubitBlochVector?
    ) {
        self.purity = purity
        self.recoveredKet = recoveredKet
        self.qubitBlochVector = qubitBlochVector
    }
}

public enum OperatorSemantics: Hashable, Sendable, Codable {
    case generic(traits: Set<OperatorTrait>)
    case density(DensityOperatorSummary)
}
