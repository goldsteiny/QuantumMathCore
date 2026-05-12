import Foundation

public extension Matrix where Element == Scalar {
    func row(_ index: Int) -> [Scalar] {
        Array(values[(index * cols)..<((index + 1) * cols)])
    }

    func transposed() -> Matrix<Scalar> {
        var transposedValues = Array(repeating: Scalar.zero, count: rows * cols)
        for row in 0..<rows {
            for col in 0..<cols {
                transposedValues[(col * rows) + row] = self[row, col]
            }
        }
        return Matrix(uncheckedRows: cols, cols: rows, values: transposedValues)
    }

    func trace() -> Scalar? {
        guard isSquare else {
            return nil
        }
        return (0..<rows).reduce(.zero) { partial, index in
            partial + self[index, index]
        }
    }

    func scaled(by scalar: Scalar) -> Matrix<Scalar> {
        Matrix(uncheckedRows: rows, cols: cols, values: values.map { $0 * scalar })
    }

    func adding(_ other: Matrix<Scalar>) throws -> Matrix<Scalar> {
        guard rows == other.rows, cols == other.cols else {
            throw QuantumMathError.operationNotDefined("Matrix dimensions must align.")
        }
        return Matrix(
            uncheckedRows: rows,
            cols: cols,
            values: zip(values, other.values).map { $0.0 + $0.1 }
        )
    }

    func subtracting(_ other: Matrix<Scalar>) throws -> Matrix<Scalar> {
        guard rows == other.rows, cols == other.cols else {
            throw QuantumMathError.operationNotDefined("Matrix dimensions must align.")
        }
        return Matrix(
            uncheckedRows: rows,
            cols: cols,
            values: zip(values, other.values).map { $0.0 - $0.1 }
        )
    }

    func tensorProduct(with other: Matrix<Scalar>) -> Matrix<Scalar> {
        let newRows = rows * other.rows
        let newCols = cols * other.cols
        var result = Matrix.zero(rows: newRows, cols: newCols)
        for leftRow in 0..<rows {
            for leftCol in 0..<cols {
                for rightRow in 0..<other.rows {
                    for rightCol in 0..<other.cols {
                        let row = (leftRow * other.rows) + rightRow
                        let col = (leftCol * other.cols) + rightCol
                        result[row, col] = self[leftRow, leftCol] * other[rightRow, rightCol]
                    }
                }
            }
        }
        return result
    }

    func isIdentity(tolerance: Double) -> Bool {
        guard isSquare else {
            return false
        }
        return approximatelyEquals(.identity(size: rows), epsilon: tolerance)
    }

    func isIdempotent(tolerance: Double) -> Bool {
        guard isSquare, let squared = try? multiplied(by: self) else {
            return false
        }
        return squared.approximatelyEquals(self, epsilon: tolerance)
    }

    func isUnitary(tolerance: Double) -> Bool {
        guard isSquare,
              let product = try? conjugateTransposed().multiplied(by: self) else {
            return false
        }
        return product.isIdentity(tolerance: tolerance)
    }

    func isPositiveSemidefinite(tolerance: Double) -> Bool {
        guard isHermitian(tolerance: tolerance) else {
            return false
        }

        // Gershgorin-style lower bound. This is intentionally conservative:
        // values that cannot be certified here can still be handled by callers
        // using a spectral analyzer.
        for row in 0..<rows {
            let diagonal = self[row, row].approximateValue
            guard abs(diagonal.im) <= tolerance else {
                return false
            }
            let radius = (0..<cols).reduce(0.0) { partial, col in
                guard col != row else {
                    return partial
                }
                return partial + Foundation.sqrt(self[row, col].magnitudeSquaredApproximate)
            }
            if diagonal.re - radius < -tolerance {
                return false
            }
        }
        return true
    }
}
