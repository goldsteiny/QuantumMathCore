import Foundation

public struct HermitianEigenDecompositionRaw: Hashable, Sendable, Codable {
    public let dimension: Int
    public let eigenvalues: [Double]
    public let eigenvectors: [ComplexNumber]

    public init(
        dimension: Int,
        eigenvalues: [Double],
        eigenvectors: [ComplexNumber]
    ) {
        self.dimension = dimension
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
    }

    public func eigenvector(at column: Int) -> [ComplexNumber] {
        (0..<dimension).map { row in
            eigenvectors[(row * dimension) + column]
        }
    }
}

public struct SingularValueDecompositionRaw: Hashable, Sendable, Codable {
    public let rows: Int
    public let cols: Int
    public let rank: Int
    public let singularValues: [Double]
    public let leftVectors: [ComplexNumber]
    public let rightVectors: [ComplexNumber]

    public init(
        rows: Int,
        cols: Int,
        rank: Int,
        singularValues: [Double],
        leftVectors: [ComplexNumber],
        rightVectors: [ComplexNumber]
    ) {
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.singularValues = singularValues
        self.leftVectors = leftVectors
        self.rightVectors = rightVectors
    }

    public func leftVector(at column: Int) -> [ComplexNumber] {
        (0..<rows).map { row in
            leftVectors[(row * rank) + column]
        }
    }

    public func rightVector(at column: Int) -> [ComplexNumber] {
        (0..<cols).map { row in
            rightVectors[(row * rank) + column]
        }
    }
}

public protocol HermitianEigenSolver: Sendable {
    func decompose(_ matrix: Matrix<Scalar>) throws -> HermitianEigenDecompositionRaw
}

public protocol SingularValueSolver: Sendable {
    func decompose(_ matrix: Matrix<Scalar>) throws -> SingularValueDecompositionRaw
}
