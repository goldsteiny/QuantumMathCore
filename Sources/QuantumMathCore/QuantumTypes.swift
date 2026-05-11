import Foundation

public struct AtomicSpace: Hashable, Sendable, Codable {
    public let dimension: Int

    public init(dimension: Int) throws {
        guard dimension >= 2 else {
            throw QuantumMathError.operationNotDefined("Atomic spaces require dimension >= 2.")
        }
        self.dimension = dimension
    }

    public static let qubit = try! AtomicSpace(dimension: 2)
    public static let qutrit = try! AtomicSpace(dimension: 3)
    public static let c4 = try! AtomicSpace(dimension: 4)
}

public enum Space: Hashable, Sendable, Codable {
    case atomic(AtomicSpace)
    case tensor([AtomicSpace])

    public init(validating factors: [AtomicSpace], maxComputableDimension: Int) throws {
        guard !factors.isEmpty else {
            throw QuantumMathError.operationNotDefined("Space requires at least one factor.")
        }
        let totalDimension = factors.map(\.dimension).reduce(1, *)
        guard totalDimension <= maxComputableDimension else {
            throw QuantumMathError.operationNotDefined(
                "Total dimension exceeds configured maximum \(maxComputableDimension)."
            )
        }
        if factors.count == 1, let factor = factors.first {
            self = .atomic(factor)
        } else {
            self = .tensor(factors)
        }
    }

    public var factors: [AtomicSpace] {
        switch self {
        case let .atomic(space):
            return [space]
        case let .tensor(factors):
            return factors
        }
    }

    public var dimension: Int {
        factors.map(\.dimension).reduce(1, *)
    }

    public func isCoordinateCompatible(with other: Space) -> Bool {
        dimension == other.dimension
    }
}

public enum AtomicBasisKind: String, Hashable, Sendable, Codable {
    case computational
    case hadamard
}

public struct Basis: Hashable, Sendable, Codable {
    public let space: Space
    public let factorKinds: [AtomicBasisKind]

    public init(space: Space, factorKinds: [AtomicBasisKind]) throws {
        guard factorKinds.count == space.factors.count else {
            throw QuantumMathError.operationNotDefined("Basis factors must match space factors.")
        }
        self.space = space
        self.factorKinds = factorKinds
    }

    public static func computational(for space: Space) -> Basis {
        try! Basis(
            space: space,
            factorKinds: Array(repeating: .computational, count: space.factors.count)
        )
    }
}

public struct Ket: Hashable, Sendable, Codable {
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

public struct Operator: Hashable, Sendable, Codable {
    public let domain: Space
    public let codomain: Space
    public let columnBasis: Basis
    public let rowBasis: Basis
    public let entries: Matrix<Scalar>

    public init(
        domain: Space,
        codomain: Space,
        columnBasis: Basis,
        rowBasis: Basis,
        entries: Matrix<Scalar>
    ) throws {
        guard columnBasis.space == domain else {
            throw QuantumMathError.incompatibleBases(lhs: columnBasis, rhs: .computational(for: domain))
        }
        guard rowBasis.space == codomain else {
            throw QuantumMathError.incompatibleBases(lhs: rowBasis, rhs: .computational(for: codomain))
        }
        guard entries.rows == codomain.dimension, entries.cols == domain.dimension else {
            throw QuantumMathError.invalidMatrixShape(
                expectedRows: codomain.dimension,
                expectedCols: domain.dimension,
                actualRows: entries.rows,
                actualCols: entries.cols
            )
        }
        self.domain = domain
        self.codomain = codomain
        self.columnBasis = columnBasis
        self.rowBasis = rowBasis
        self.entries = entries
    }
}

public struct EigenComponent: Hashable, Sendable, Codable {
    public let eigenvalue: Scalar
    public let multiplicity: Int
    public let eigenvectors: [Ket]

    public init(eigenvalue: Scalar, multiplicity: Int, eigenvectors: [Ket]) {
        self.eigenvalue = eigenvalue
        self.multiplicity = multiplicity
        self.eigenvectors = eigenvectors
    }
}

public struct SpectralDecomposition: Hashable, Sendable, Codable {
    public let space: Space
    public let basis: Basis
    public let components: [EigenComponent]

    public init(space: Space, basis: Basis, components: [EigenComponent]) {
        self.space = space
        self.basis = basis
        self.components = components
    }
}

public struct SingularValueComponent: Hashable, Sendable, Codable {
    public let singularValue: Scalar
    public let leftVector: Ket
    public let rightVector: Ket

    public init(singularValue: Scalar, leftVector: Ket, rightVector: Ket) {
        self.singularValue = singularValue
        self.leftVector = leftVector
        self.rightVector = rightVector
    }
}

public struct SingularValueDecomposition: Hashable, Sendable, Codable {
    public let domain: Space
    public let codomain: Space
    public let columnBasis: Basis
    public let rowBasis: Basis
    public let components: [SingularValueComponent]

    public init(
        domain: Space,
        codomain: Space,
        columnBasis: Basis,
        rowBasis: Basis,
        components: [SingularValueComponent]
    ) {
        self.domain = domain
        self.codomain = codomain
        self.columnBasis = columnBasis
        self.rowBasis = rowBasis
        self.components = components
    }
}

public enum QuantumMathError: Error, Hashable, Sendable, Codable {
    case incompatibleSpaces(expected: Space, actual: Space)
    case incompatibleBases(lhs: Basis, rhs: Basis)
    case invalidCoefficientCount(expected: Int, actual: Int)
    case invalidMatrixShape(expectedRows: Int, expectedCols: Int, actualRows: Int, actualCols: Int)
    case unsupportedExactOperation(String)
    case operationNotDefined(String)
    case backendFailure(BackendFailure)
}

public struct BackendFailure: Hashable, Sendable, Codable {
    public enum Kind: String, Hashable, Sendable, Codable {
        case invalidInput
        case workspaceQueryFailed
        case factorizationFailed
        case bufferDecodeFailed
        case nonFiniteOutput
        case unknown
    }

    public let backend: String
    public let kind: Kind
    public let message: String

    public init(backend: String, kind: Kind, message: String) {
        self.backend = backend
        self.kind = kind
        self.message = message
    }
}
