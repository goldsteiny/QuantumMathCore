import Foundation

public enum QuantumDomain {
    public static func dagger(_ ket: Ket) throws -> Bra {
        try Bra(
            space: ket.space,
            basis: ket.basis,
            coefficients: ket.coefficients.map(\.conjugated)
        )
    }

    public static func dagger(_ bra: Bra) throws -> Ket {
        try Ket(
            space: bra.space,
            basis: bra.basis,
            coefficients: bra.coefficients.map(\.conjugated)
        )
    }

    public static func applyOperator(_ op: Operator, _ ket: Ket) throws -> Ket {
        guard op.domain.isCoordinateCompatible(with: ket.space) else {
            throw QuantumMathError.incompatibleSpaces(expected: op.domain, actual: ket.space)
        }
        guard op.columnBasis == ket.basis else {
            throw QuantumMathError.incompatibleBases(lhs: op.columnBasis, rhs: ket.basis)
        }
        let coefficients = try op.entries.multiplied(by: ket.coefficients)
        return try Ket(
            space: op.codomain,
            basis: op.rowBasis,
            coefficients: coefficients
        )
    }

    public static func compose(_ lhs: Operator, _ rhs: Operator) throws -> Operator {
        guard lhs.domain.isCoordinateCompatible(with: rhs.codomain) else {
            throw QuantumMathError.incompatibleSpaces(expected: lhs.domain, actual: rhs.codomain)
        }
        guard lhs.columnBasis == rhs.rowBasis else {
            throw QuantumMathError.incompatibleBases(lhs: lhs.columnBasis, rhs: rhs.rowBasis)
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

    public static func outerProduct(_ ket: Ket, _ bra: Bra) throws -> Operator {
        guard ket.space.isCoordinateCompatible(with: bra.space) else {
            throw QuantumMathError.incompatibleSpaces(expected: ket.space, actual: bra.space)
        }
        guard ket.basis == bra.basis else {
            throw QuantumMathError.incompatibleBases(lhs: ket.basis, rhs: bra.basis)
        }
        guard ket.coefficients.count == bra.coefficients.count else {
            throw QuantumMathError.invalidCoefficientCount(
                expected: ket.coefficients.count,
                actual: bra.coefficients.count
            )
        }
        let dimension = ket.space.dimension
        let values = (0..<dimension).flatMap { row in
            (0..<dimension).map { col in
                ket.coefficients[row] * bra.coefficients[col]
            }
        }
        return try Operator(
            domain: bra.space,
            codomain: ket.space,
            columnBasis: bra.basis,
            rowBasis: ket.basis,
            entries: Matrix(uncheckedRows: dimension, cols: dimension, values: values)
        )
    }

    public static func outerProduct(_ ket: Ket, _ braCoefficients: [Scalar]) throws -> Operator {
        try outerProduct(
            ket,
            Bra(space: ket.space, basis: ket.basis, coefficients: braCoefficients)
        )
    }

    public static func innerProduct(_ lhs: Ket, _ rhs: Ket) throws -> Scalar {
        guard lhs.space.isCoordinateCompatible(with: rhs.space) else {
            throw QuantumMathError.incompatibleSpaces(expected: lhs.space, actual: rhs.space)
        }
        guard lhs.basis == rhs.basis else {
            throw QuantumMathError.incompatibleBases(lhs: lhs.basis, rhs: rhs.basis)
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
