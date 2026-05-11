import Foundation

public enum QuantumDomain {
    public static func dagger(_ ket: Ket) throws -> [Scalar] {
        ket.coefficients.map(\.conjugated)
    }

    public static func applyOperator(_ op: Operator, _ ket: Ket) throws -> Ket {
        guard op.domain.isCoordinateCompatible(with: ket.space),
              op.columnBasis == ket.basis else {
            throw QuantumMathError.incompatibleSpaces(expected: op.domain, actual: ket.space)
        }
        let coefficients = try op.entries.multiplied(by: ket.coefficients)
        return try Ket(
            space: op.codomain,
            basis: op.rowBasis,
            coefficients: coefficients
        )
    }

    public static func compose(_ lhs: Operator, _ rhs: Operator) throws -> Operator {
        guard lhs.domain.isCoordinateCompatible(with: rhs.codomain),
              lhs.columnBasis == rhs.rowBasis else {
            throw QuantumMathError.operationNotDefined("Operator composition requires compatible spaces and bases.")
        }
        let entries = try lhs.entries.multiplied(by: rhs.entries)
        return try Operator(
            domain: rhs.domain,
            codomain: lhs.codomain,
            columnBasis: rhs.columnBasis,
            rowBasis: lhs.rowBasis,
            entries: entries
        )
    }

    public static func outerProduct(_ ket: Ket, _ bra: [Scalar]) throws -> Operator {
        guard ket.coefficients.count == bra.count else {
            throw QuantumMathError.invalidCoefficientCount(
                expected: ket.coefficients.count,
                actual: bra.count
            )
        }
        let dimension = ket.space.dimension
        let values = (0..<dimension).flatMap { row in
            (0..<dimension).map { col in
                ket.coefficients[row] * bra[col]
            }
        }
        return try Operator(
            domain: ket.space,
            codomain: ket.space,
            columnBasis: ket.basis,
            rowBasis: ket.basis,
            entries: Matrix(uncheckedRows: dimension, cols: dimension, values: values)
        )
    }

    public static func innerProduct(_ lhs: Ket, _ rhs: Ket) throws -> Scalar {
        guard lhs.space.isCoordinateCompatible(with: rhs.space),
              lhs.basis == rhs.basis else {
            throw QuantumMathError.incompatibleSpaces(expected: lhs.space, actual: rhs.space)
        }
        return zip(lhs.coefficients, rhs.coefficients).reduce(.zero) { partial, pair in
            partial + (pair.0.conjugated * pair.1)
        }
    }

    public static func normSquared(_ ket: Ket) -> Double {
        ket.coefficients.reduce(0) { partial, coefficient in
            partial + coefficient.magnitudeSquaredApproximate
        }
    }
}
