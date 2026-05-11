import Foundation

public struct Matrix<Element: Hashable & Sendable & Codable>: Hashable, Sendable, Codable {
    public let rows: Int
    public let cols: Int
    public private(set) var values: [Element]

    public init(rows: Int, cols: Int, values: [Element]) throws {
        guard rows >= 0, cols >= 0 else {
            throw QuantumMathError.operationNotDefined("Matrix dimensions must be non-negative.")
        }
        guard values.count == rows * cols else {
            throw QuantumMathError.invalidMatrixShape(
                expectedRows: rows,
                expectedCols: cols,
                actualRows: rows,
                actualCols: values.count
            )
        }
        self.rows = rows
        self.cols = cols
        self.values = values
    }

    init(uncheckedRows rows: Int, cols: Int, values: [Element]) {
        self.rows = rows
        self.cols = cols
        self.values = values
    }

    public subscript(row: Int, col: Int) -> Element {
        get { values[(row * cols) + col] }
        set { values[(row * cols) + col] = newValue }
    }
}

public extension Matrix where Element == Scalar {
    static func zero(rows: Int, cols: Int) -> Matrix<Scalar> {
        Matrix(uncheckedRows: rows, cols: cols, values: Array(repeating: .zero, count: rows * cols))
    }

    static func identity(size: Int) -> Matrix<Scalar> {
        var matrix = Matrix.zero(rows: size, cols: size)
        for index in 0..<size {
            matrix[index, index] = .one
        }
        return matrix
    }

    var isSquare: Bool {
        rows == cols
    }

    func conjugateTransposed() -> Matrix<Scalar> {
        var transposed = Array(repeating: Scalar.zero, count: rows * cols)
        for row in 0..<rows {
            for col in 0..<cols {
                transposed[(col * rows) + row] = self[row, col].conjugated
            }
        }
        return Matrix(uncheckedRows: cols, cols: rows, values: transposed)
    }

    func multiplied(by vector: [Scalar]) throws -> [Scalar] {
        guard cols == vector.count else {
            throw QuantumMathError.operationNotDefined("Matrix/vector dimensions must align.")
        }
        return (0..<rows).map { row in
            (0..<cols).reduce(.zero) { partial, col in
                partial + (self[row, col] * vector[col])
            }
        }
    }

    func multiplied(by other: Matrix<Scalar>) throws -> Matrix<Scalar> {
        guard cols == other.rows else {
            throw QuantumMathError.operationNotDefined("Matrix dimensions must align for multiplication.")
        }

        var result = Matrix.zero(rows: rows, cols: other.cols)
        for row in 0..<rows {
            for col in 0..<other.cols {
                result[row, col] = (0..<cols).reduce(.zero) { partial, index in
                    partial + (self[row, index] * other[index, col])
                }
            }
        }
        return result
    }

    func approximatelyEquals(_ other: Matrix<Scalar>, epsilon: Double) -> Bool {
        guard rows == other.rows, cols == other.cols else {
            return false
        }
        return zip(values, other.values).allSatisfy { lhs, rhs in
            lhs.approximatelyEquals(rhs, epsilon: epsilon)
        }
    }

    func isDiagonal(tolerance: Double) -> Bool {
        for row in 0..<rows {
            for col in 0..<cols where row != col {
                if !self[row, col].isZero(epsilon: tolerance) {
                    return false
                }
            }
        }
        return true
    }

    func isHermitian(tolerance: Double) -> Bool {
        guard isSquare else {
            return false
        }
        return approximatelyEquals(conjugateTransposed(), epsilon: tolerance)
    }
}
