import Foundation
import ExactValueRecoveryCore

public extension Basis {
    var isComputationalCoordinateBasis: Bool {
        factorKinds.allSatisfy { $0 == .computational }
    }

    func isCoordinateCompatible(with other: Basis) -> Bool {
        self == other
            || (
                space.isCoordinateCompatible(with: other.space)
                    && isComputationalCoordinateBasis
                    && other.isComputationalCoordinateBasis
            )
    }
}

public extension QuantumDomain {
    static func dagger(_ operatorValue: Operator) throws -> Operator {
        try Operator(
            domain: operatorValue.codomain,
            codomain: operatorValue.domain,
            columnBasis: operatorValue.rowBasis,
            rowBasis: operatorValue.columnBasis,
            entries: operatorValue.entries.conjugateTransposed()
        )
    }

    static func dagger(_ value: QuantumValue) throws -> QuantumValue {
        switch value {
        case let .ket(ket):
            return .bra(try dagger(ket))
        case let .bra(bra):
            return .ket(try dagger(bra))
        case let .oper(operatorValue):
            return .oper(try dagger(operatorValue))
        case .scalar:
            throw QuantumMathError.operationNotDefined(
                "Dagger is only defined for bras, kets, and operators."
            )
        }
    }

    static func transpose(_ operatorValue: Operator) throws -> Operator {
        try Operator(
            domain: operatorValue.codomain,
            codomain: operatorValue.domain,
            columnBasis: operatorValue.rowBasis,
            rowBasis: operatorValue.columnBasis,
            entries: operatorValue.entries.transposed()
        )
    }

    static func trace(_ operatorValue: Operator) throws -> Scalar {
        guard operatorValue.domain == operatorValue.codomain,
              operatorValue.rowBasis == operatorValue.columnBasis,
              let trace = operatorValue.entries.trace() else {
            throw QuantumMathError.operationNotDefined(
                "Trace requires a square operator with matching basis on rows and columns."
            )
        }
        return trace
    }

    static func innerProduct(_ bra: Bra, _ ket: Ket) throws -> Scalar {
        guard bra.space.isCoordinateCompatible(with: ket.space) else {
            throw QuantumMathError.incompatibleSpaces(expected: bra.space, actual: ket.space)
        }
        guard bra.basis.isCoordinateCompatible(with: ket.basis) else {
            throw QuantumMathError.incompatibleBases(lhs: bra.basis, rhs: ket.basis)
        }
        return zip(bra.coefficients, ket.coefficients).reduce(.zero) { partial, pair in
            partial + (pair.0 * pair.1)
        }
    }

    static func applyBraAfterOperator(_ bra: Bra, _ operatorValue: Operator) throws -> Bra {
        let daggerBra = try dagger(bra)
        let appliedKet = try applyOperator(try dagger(operatorValue), daggerBra)
        return try dagger(appliedKet)
    }

    static func sandwich(_ bra: Bra, _ operatorValue: Operator, _ ket: Ket) throws -> Scalar {
        try innerProduct(bra, applyOperator(operatorValue, ket))
    }

    static func tensor(
        _ lhs: Ket,
        _ rhs: Ket,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Ket {
        let space = try tensorSpace(lhs.space, rhs.space, config: config)
        let basis = try Basis(
            space: space,
            factorKinds: lhs.basis.factorKinds + rhs.basis.factorKinds
        )
        return try Ket(
            space: space,
            basis: basis,
            coefficients: lhs.coefficients.flatMap { lhsScalar in
                rhs.coefficients.map { rhsScalar in lhsScalar * rhsScalar }
            }
        )
    }

    static func tensor(
        _ lhs: Bra,
        _ rhs: Bra,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Bra {
        let space = try tensorSpace(lhs.space, rhs.space, config: config)
        let basis = try Basis(
            space: space,
            factorKinds: lhs.basis.factorKinds + rhs.basis.factorKinds
        )
        return try Bra(
            space: space,
            basis: basis,
            coefficients: lhs.coefficients.flatMap { lhsScalar in
                rhs.coefficients.map { rhsScalar in lhsScalar * rhsScalar }
            }
        )
    }

    static func tensor(
        _ lhs: Operator,
        _ rhs: Operator,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        guard lhs.domain == lhs.codomain, rhs.domain == rhs.codomain else {
            throw QuantumMathError.operationNotDefined("Tensoring non-square operators is not supported.")
        }
        let domain = try tensorSpace(lhs.domain, rhs.domain, config: config)
        let codomain = try tensorSpace(lhs.codomain, rhs.codomain, config: config)
        return try Operator(
            domain: domain,
            codomain: codomain,
            columnBasis: try Basis(
                space: domain,
                factorKinds: lhs.columnBasis.factorKinds + rhs.columnBasis.factorKinds
            ),
            rowBasis: try Basis(
                space: codomain,
                factorKinds: lhs.rowBasis.factorKinds + rhs.rowBasis.factorKinds
            ),
            entries: lhs.entries.tensorProduct(with: rhs.entries)
        )
    }

    static func normalize(_ ket: Ket, config: QuantumMathConfig = .ketStepsDefault) throws -> Ket {
        try Ket(
            space: ket.space,
            basis: ket.basis,
            coefficients: normalizedCoefficients(ket.coefficients, config: config)
        )
    }

    static func normalize(_ bra: Bra, config: QuantumMathConfig = .ketStepsDefault) throws -> Bra {
        try Bra(
            space: bra.space,
            basis: bra.basis,
            coefficients: normalizedCoefficients(bra.coefficients, config: config)
        )
    }

    static func normalize(_ operatorValue: Operator) throws -> Operator {
        let trace = try trace(operatorValue)
        guard !trace.isZero(epsilon: 0) else {
            throw QuantumMathError.operationNotDefined(
                "Operators with zero trace cannot be normalized to tr(A) = 1."
            )
        }
        return try operatorDividing(operatorValue, by: trace)
    }

    static func scale(_ scalar: Scalar, value: QuantumValue) throws -> QuantumValue {
        switch value {
        case let .scalar(otherScalar):
            return .scalar(scalar * otherScalar)
        case let .ket(ket):
            return .ket(try Ket(
                space: ket.space,
                basis: ket.basis,
                coefficients: ket.coefficients.map { scalar * $0 }
            ))
        case let .bra(bra):
            return .bra(try Bra(
                space: bra.space,
                basis: bra.basis,
                coefficients: bra.coefficients.map { scalar * $0 }
            ))
        case let .oper(operatorValue):
            return .oper(try Operator(
                domain: operatorValue.domain,
                codomain: operatorValue.codomain,
                columnBasis: operatorValue.columnBasis,
                rowBasis: operatorValue.rowBasis,
                entries: operatorValue.entries.scaled(by: scalar)
            ))
        }
    }

    static func add(_ lhs: QuantumValue, _ rhs: QuantumValue) throws -> QuantumValue {
        try combine(lhs, rhs, transform: +)
    }

    static func subtract(_ lhs: QuantumValue, _ rhs: QuantumValue) throws -> QuantumValue {
        try combine(lhs, rhs, transform: -)
    }

    static func weightedSum(
        terms: [(weight: Scalar, value: QuantumValue)],
        normalization: WeightedSumNormalization,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> QuantumValue {
        guard terms.count >= 2, let first = terms.first?.value else {
            throw QuantumMathError.operationNotDefined("Weighted sum requires at least two terms.")
        }

        switch first {
        case .scalar:
            throw QuantumMathError.operationNotDefined(
                "Weighted sum is only defined for kets, bras, and operators."
            )
        case let .ket(firstKet):
            let result = try weightedKetSum(firstKet: firstKet, terms: terms)
            switch normalization {
            case .none:
                return .ket(result)
            case .stateUnitInnerProduct:
                return .ket(try normalize(result, config: config))
            case .operatorUnitTrace:
                throw QuantumMathError.operationNotDefined(
                    "Trace normalization is only defined for operators."
                )
            }
        case let .bra(firstBra):
            let result = try weightedBraSum(firstBra: firstBra, terms: terms)
            switch normalization {
            case .none:
                return .bra(result)
            case .stateUnitInnerProduct:
                return .bra(try normalize(result, config: config))
            case .operatorUnitTrace:
                throw QuantumMathError.operationNotDefined(
                    "Trace normalization is only defined for operators."
                )
            }
        case let .oper(firstOperator):
            let result = try weightedOperatorSum(firstOperator: firstOperator, terms: terms)
            switch normalization {
            case .none:
                return .oper(result)
            case .stateUnitInnerProduct:
                throw QuantumMathError.operationNotDefined(
                    "Unit inner-product normalization is only defined for bras and kets."
                )
            case .operatorUnitTrace:
                return .oper(try normalize(result))
            }
        }
    }

    static func pureDensityState(
        from value: QuantumValue,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        let ket = try normalizedStateKet(from: value, config: config)
        return try outerProduct(ket, dagger(ket))
    }

    static func mixedDensityState(
        terms: [(weight: Scalar, value: QuantumValue)],
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        guard terms.count >= 2 else {
            throw QuantumMathError.operationNotDefined(
                "Mixed state creation requires at least two state vectors."
            )
        }
        let weights = try normalizedMixedDensityWeights(
            terms.map(\.weight),
            config: config
        )
        let kets = try terms.map { try normalizedStateKet(from: $0.value, config: config) }
        guard let firstKet = kets.first else {
            throw QuantumMathError.operationNotDefined(
                "Mixed state creation requires at least two state vectors."
            )
        }
        try kets.dropFirst().forEach { ket in
            try requireSameSpace(firstKet.space, ket.space)
            try requireSameBasis(firstKet.basis, ket.basis)
        }

        let projectors = try zip(weights, kets).map { weight, ket in
            try outerProduct(ket, dagger(ket)).entries.scaled(by: weight)
        }
        let entries = try projectors.dropFirst().reduce(projectors[0]) { partial, next in
            try partial.adding(next)
        }
        return try Operator(
            domain: firstKet.space,
            codomain: firstKet.space,
            columnBasis: firstKet.basis,
            rowBasis: firstKet.basis,
            entries: entries
        )
    }

    static func densityOperator(
        fromBlochVector vector: QubitBlochVector,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        let epsilon = config.scalarComparisonEpsilon
        guard let x = realFiniteComponent(vector.x, epsilon: epsilon),
              let y = realFiniteComponent(vector.y, epsilon: epsilon),
              let z = realFiniteComponent(vector.z, epsilon: epsilon) else {
            throw QuantumMathError.operationNotDefined(
                "Bloch vectors require finite real x, y, and z coordinates."
            )
        }

        let radiusSquared = (x * x) + (y * y) + (z * z)
        guard radiusSquared <= 1 + epsilon else {
            throw QuantumMathError.operationNotDefined(
                "Bloch vector radius must satisfy x^2 + y^2 + z^2 <= 1."
            )
        }

        let half = Scalar(real: Rational(1, 2))
        let xScalar = Scalar.approx(ComplexNumber(re: x, im: 0))
        let yScalar = Scalar.approx(ComplexNumber(re: y, im: 0))
        let zScalar = Scalar.approx(ComplexNumber(re: z, im: 0))

        let rho00 = half * (Scalar.one + zScalar)
        let rho11 = half * (Scalar.one - zScalar)
        let rho01 = half * (xScalar - (Scalar.i * yScalar))
        let rho10 = half * (xScalar + (Scalar.i * yScalar))

        let qubitSpace = Space.atomic(.qubit)
        let basis = Basis.computational(for: qubitSpace)

        return try Operator(
            domain: qubitSpace,
            codomain: qubitSpace,
            columnBasis: basis,
            rowBasis: basis,
            entries: try Matrix(rows: 2, cols: 2, values: [rho00, rho01, rho10, rho11])
        )
    }

    static func ket(
        fromBlochAngles angles: QubitBlochAngles,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Ket {
        let epsilon = config.scalarComparisonEpsilon
        guard let theta = realFiniteComponent(angles.thetaRadians, epsilon: epsilon),
              let phi = realFiniteComponent(angles.phiRadians, epsilon: epsilon) else {
            throw QuantumMathError.operationNotDefined(
                "Bloch angles require finite real theta and phi values."
            )
        }

        let halfTheta = theta / 2
        let alpha = Scalar.approx(ComplexNumber(re: Foundation.cos(halfTheta), im: 0))
        let betaMagnitude = Foundation.sin(halfTheta)
        let beta = Scalar.approx(
            ComplexNumber(
                re: betaMagnitude * Foundation.cos(phi),
                im: betaMagnitude * Foundation.sin(phi)
            )
        )

        let qubitSpace = Space.atomic(.qubit)
        let basis = Basis.computational(for: qubitSpace)
        return try Ket(space: qubitSpace, basis: basis, coefficients: [alpha, beta])
    }

    static func bra(
        fromBlochAngles angles: QubitBlochAngles,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Bra {
        try dagger(ket(fromBlochAngles: angles, config: config))
    }

    static func blochVector(
        for ket: Ket,
        config: QuantumMathConfig = .ketStepsDefault
    ) -> QubitBlochVector? {
        guard ket.space == .atomic(.qubit),
              ket.basis == .computational(for: ket.space),
              abs(normSquared(ket) - 1) <= config.scalarComparisonEpsilon else {
            return nil
        }

        return qubitBlochVector(forNormalizedCoefficients: ket.coefficients)
    }

    static func blochVector(
        for bra: Bra,
        config: QuantumMathConfig = .ketStepsDefault
    ) -> QubitBlochVector? {
        guard let ket = try? dagger(bra) else {
            return nil
        }
        return blochVector(for: ket, config: config)
    }

    static func blochVector(
        for operatorValue: Operator,
        config: QuantumMathConfig = .ketStepsDefault
    ) -> QubitBlochVector? {
        guard case let .density(summary) = operatorSemantics(operatorValue, config: config) else {
            return nil
        }
        return summary.qubitBlochVector
    }

    static func commutator(_ lhs: Operator, _ rhs: Operator) throws -> Operator {
        let forward = try compose(lhs, rhs)
        let reverse = try compose(rhs, lhs)
        guard case let .oper(result) = try subtract(.oper(forward), .oper(reverse)) else {
            throw QuantumMathError.operationNotDefined("Expected operator result.")
        }
        return result
    }

    static func anticommutator(_ lhs: Operator, _ rhs: Operator) throws -> Operator {
        let forward = try compose(lhs, rhs)
        let reverse = try compose(rhs, lhs)
        guard case let .oper(result) = try add(.oper(forward), .oper(reverse)) else {
            throw QuantumMathError.operationNotDefined("Expected operator result.")
        }
        return result
    }

    static func partialTrace(
        _ operatorValue: Operator,
        tracing selection: PartialTraceSelection
    ) throws -> Operator {
        let source = try TensorProductEndomorphism(operatorValue)
        let resolvedSelection = try ResolvedPartialTraceSelection(selection: selection, source: source)
        let entries = try partialTraceEntries(source: source, selection: resolvedSelection)
        return try Operator(
            domain: resolvedSelection.retainedSpace,
            codomain: resolvedSelection.retainedSpace,
            columnBasis: resolvedSelection.retainedBasis,
            rowBasis: resolvedSelection.retainedBasis,
            entries: entries
        )
    }

    static func power(_ operatorValue: Operator, exponent: Int) throws -> Operator {
        guard exponent >= 0 else {
            throw QuantumMathError.operationNotDefined("Power requires a non-negative exponent.")
        }
        guard operatorValue.domain == operatorValue.codomain,
              operatorValue.rowBasis == operatorValue.columnBasis else {
            throw QuantumMathError.operationNotDefined(
                "Power requires a square operator with matching basis on rows and columns."
            )
        }

        if exponent == 0 {
            return try Operator(
                domain: operatorValue.domain,
                codomain: operatorValue.codomain,
                columnBasis: operatorValue.columnBasis,
                rowBasis: operatorValue.rowBasis,
                entries: .identity(size: operatorValue.domain.dimension)
            )
        }

        var remaining = exponent
        var result = try Operator(
            domain: operatorValue.domain,
            codomain: operatorValue.codomain,
            columnBasis: operatorValue.columnBasis,
            rowBasis: operatorValue.rowBasis,
            entries: .identity(size: operatorValue.domain.dimension)
        )
        var base = operatorValue
        while remaining > 0 {
            if remaining % 2 == 1 {
                result = try compose(result, base)
            }
            remaining /= 2
            if remaining > 0 {
                base = try compose(base, base)
            }
        }
        return result
    }

    static func convert(_ value: QuantumValue, to targetBasis: Basis) throws -> QuantumValue {
        switch value {
        case .scalar:
            throw QuantumMathError.operationNotDefined("Scalars do not have a basis.")
        case let .ket(ket):
            return .ket(try convert(ket, to: targetBasis))
        case let .bra(bra):
            return .bra(try convert(bra, to: targetBasis))
        case let .oper(operatorValue):
            guard operatorValue.domain == operatorValue.codomain else {
                throw QuantumMathError.operationNotDefined(
                    "Only square operators can be converted with a single target basis."
                )
            }
            return .oper(try convert(operatorValue, to: targetBasis))
        }
    }

    static func convert(_ ket: Ket, to targetBasis: Basis) throws -> Ket {
        try requireSameSpace(ket.space, targetBasis.space)
        guard ket.basis != targetBasis else {
            return ket
        }
        let transform = try basisTransform(from: ket.basis, to: targetBasis)
        return try Ket(
            space: ket.space,
            basis: targetBasis,
            coefficients: try transform.multiplied(by: ket.coefficients)
        )
    }

    static func convert(_ bra: Bra, to targetBasis: Basis) throws -> Bra {
        try dagger(convert(try dagger(bra), to: targetBasis))
    }

    static func convert(_ operatorValue: Operator, to targetBasis: Basis) throws -> Operator {
        try requireSameSpace(operatorValue.domain, targetBasis.space)
        try requireSameSpace(operatorValue.codomain, targetBasis.space)
        let rowTransform = try basisTransform(from: operatorValue.rowBasis, to: targetBasis)
        let columnTransform = try basisTransform(from: operatorValue.columnBasis, to: targetBasis)
        return try Operator(
            domain: operatorValue.domain,
            codomain: operatorValue.codomain,
            columnBasis: targetBasis,
            rowBasis: targetBasis,
            entries: try rowTransform
                .multiplied(by: operatorValue.entries)
                .multiplied(by: columnTransform.conjugateTransposed())
        )
    }

    static func basisTransform(from: Basis, to: Basis) throws -> Matrix<Scalar> {
        try requireSameSpace(from.space, to.space)
        guard from != to else {
            return .identity(size: from.space.dimension)
        }
        let factorMatrices = try zip(from.space.factors, zip(from.factorKinds, to.factorKinds)).map { pair in
            try factorTransform(for: pair.0, from: pair.1.0, to: pair.1.1)
        }
        guard let first = factorMatrices.first else {
            return .identity(size: from.space.dimension)
        }
        return factorMatrices.dropFirst().reduce(first) { partial, next in
            partial.tensorProduct(with: next)
        }
    }

    static func deriveTraits(from entries: Matrix<Scalar>, config: QuantumMathConfig) -> Set<OperatorTrait> {
        var traits: Set<OperatorTrait> = []
        if entries.isDiagonal(tolerance: config.matrixTraitEpsilon) {
            traits.insert(.diagonal)
        }
        if entries.isIdentity(tolerance: config.matrixTraitEpsilon) {
            traits.insert(.identity)
        }
        if entries.isHermitian(tolerance: config.matrixTraitEpsilon) {
            traits.insert(.hermitian)
        }
        if entries.isUnitary(tolerance: config.matrixTraitEpsilon) {
            traits.insert(.unitary)
        }
        if entries.isHermitian(tolerance: config.matrixTraitEpsilon),
           entries.isIdempotent(tolerance: config.matrixTraitEpsilon) {
            traits.insert(.projector)
        }
        return traits
    }

    static func operatorSemantics(
        _ operatorValue: Operator,
        config: QuantumMathConfig
    ) -> OperatorSemantics {
        guard let densitySummary = densityOperatorSummary(for: operatorValue, config: config) else {
            return .generic(traits: deriveTraits(from: operatorValue.entries, config: config))
        }
        return .density(densitySummary)
    }
}

private extension QuantumDomain {
    static func tensorSpace(
        _ lhs: Space,
        _ rhs: Space,
        config: QuantumMathConfig
    ) throws -> Space {
        let factors = lhs.factors + rhs.factors
        return try Space(validating: factors, maxComputableDimension: config.maxComputableDimension)
    }

    static func factorTransform(
        for factor: AtomicSpace,
        from: AtomicBasisKind,
        to: AtomicBasisKind
    ) throws -> Matrix<Scalar> {
        guard from != to else {
            return .identity(size: factor.dimension)
        }
        switch (factor.dimension, from, to) {
        case (2, .computational, .hadamard), (2, .hadamard, .computational):
            let inverseRootTwo = Scalar.exact(
                .canonicalSignedRationalTimesSqrt(coefficient: Rational(1, 2), radicand: 2)
            )
            return try Matrix(
                rows: 2,
                cols: 2,
                values: [
                    inverseRootTwo, inverseRootTwo,
                    inverseRootTwo, -inverseRootTwo
                ]
            )
        default:
            throw QuantumMathError.operationNotDefined("Unsupported basis conversion.")
        }
    }

    static func combine(
        _ lhs: QuantumValue,
        _ rhs: QuantumValue,
        transform: (Scalar, Scalar) -> Scalar
    ) throws -> QuantumValue {
        switch (lhs, rhs) {
        case let (.scalar(left), .scalar(right)):
            return .scalar(transform(left, right))
        case let (.ket(left), .ket(right)):
            try requireSameSpace(left.space, right.space)
            try requireSameBasis(left.basis, right.basis)
            return .ket(try Ket(
                space: left.space,
                basis: left.basis,
                coefficients: zip(left.coefficients, right.coefficients).map { transform($0.0, $0.1) }
            ))
        case let (.bra(left), .bra(right)):
            try requireSameSpace(left.space, right.space)
            try requireSameBasis(left.basis, right.basis)
            return .bra(try Bra(
                space: left.space,
                basis: left.basis,
                coefficients: zip(left.coefficients, right.coefficients).map { transform($0.0, $0.1) }
            ))
        case let (.oper(left), .oper(right)):
            try requireSameSpace(left.domain, right.domain)
            try requireSameSpace(left.codomain, right.codomain)
            try requireSameBasis(left.columnBasis, right.columnBasis)
            try requireSameBasis(left.rowBasis, right.rowBasis)
            return .oper(try Operator(
                domain: left.domain,
                codomain: left.codomain,
                columnBasis: left.columnBasis,
                rowBasis: left.rowBasis,
                entries: try Matrix(
                    rows: left.entries.rows,
                    cols: left.entries.cols,
                    values: zip(left.entries.values, right.entries.values).map {
                        transform($0.0, $0.1)
                    }
                )
            ))
        default:
            throw QuantumMathError.operationNotDefined(
                "Addition and subtraction require matching object kinds."
            )
        }
    }

    static func weightedKetSum(
        firstKet: Ket,
        terms: [(weight: Scalar, value: QuantumValue)]
    ) throws -> Ket {
        let coefficients = try terms.reduce(
            Array(repeating: Scalar.zero, count: firstKet.coefficients.count)
        ) { partial, term in
            guard case let .ket(ket) = term.value else {
                throw QuantumMathError.operationNotDefined("Weighted sum requires matching object kinds.")
            }
            try requireSameSpace(firstKet.space, ket.space)
            try requireSameBasis(firstKet.basis, ket.basis)
            return zip(partial, ket.coefficients).map { $0.0 + (term.weight * $0.1) }
        }
        return try Ket(space: firstKet.space, basis: firstKet.basis, coefficients: coefficients)
    }

    static func weightedBraSum(
        firstBra: Bra,
        terms: [(weight: Scalar, value: QuantumValue)]
    ) throws -> Bra {
        let coefficients = try terms.reduce(
            Array(repeating: Scalar.zero, count: firstBra.coefficients.count)
        ) { partial, term in
            guard case let .bra(bra) = term.value else {
                throw QuantumMathError.operationNotDefined("Weighted sum requires matching object kinds.")
            }
            try requireSameSpace(firstBra.space, bra.space)
            try requireSameBasis(firstBra.basis, bra.basis)
            return zip(partial, bra.coefficients).map { $0.0 + (term.weight * $0.1) }
        }
        return try Bra(space: firstBra.space, basis: firstBra.basis, coefficients: coefficients)
    }

    static func weightedOperatorSum(
        firstOperator: Operator,
        terms: [(weight: Scalar, value: QuantumValue)]
    ) throws -> Operator {
        let values = try terms.reduce(
            Array(repeating: Scalar.zero, count: firstOperator.entries.values.count)
        ) { partial, term in
            guard case let .oper(operatorValue) = term.value else {
                throw QuantumMathError.operationNotDefined("Weighted sum requires matching object kinds.")
            }
            try requireSameSpace(firstOperator.domain, operatorValue.domain)
            try requireSameSpace(firstOperator.codomain, operatorValue.codomain)
            try requireSameBasis(firstOperator.columnBasis, operatorValue.columnBasis)
            try requireSameBasis(firstOperator.rowBasis, operatorValue.rowBasis)
            return zip(partial, operatorValue.entries.values).map { $0.0 + (term.weight * $0.1) }
        }
        return try Operator(
            domain: firstOperator.domain,
            codomain: firstOperator.codomain,
            columnBasis: firstOperator.columnBasis,
            rowBasis: firstOperator.rowBasis,
            entries: try Matrix(
                rows: firstOperator.entries.rows,
                cols: firstOperator.entries.cols,
                values: values
            )
        )
    }

    static func partialTraceEntries(
        source: TensorProductEndomorphism,
        selection: ResolvedPartialTraceSelection
    ) throws -> Matrix<Scalar> {
        let retainedDimensions = selection.retainedOffsets.map { source.factorDimensions[$0.rawValue] }
        let tracedDimensions = selection.tracedOffsets.map { source.factorDimensions[$0.rawValue] }
        let retainedDimension = retainedDimensions.reduce(1, *)
        var result = Matrix<Scalar>.zero(rows: retainedDimension, cols: retainedDimension)
        let tracedAssignments = try tensorIndexAssignments(for: tracedDimensions)

        for retainedRow in 0..<retainedDimension {
            let retainedRowIndices = try TensorIndexing.unflatten(
                index: retainedRow,
                factorDimensions: retainedDimensions
            )
            for retainedCol in 0..<retainedDimension {
                let retainedColIndices = try TensorIndexing.unflatten(
                    index: retainedCol,
                    factorDimensions: retainedDimensions
                )
                result[retainedRow, retainedCol] = try tracedAssignments.reduce(.zero) { partial, tracedIndices in
                    let sourceRow = try sourceFlatIndex(
                        retainedIndices: retainedRowIndices,
                        tracedIndices: tracedIndices,
                        source: source,
                        selection: selection
                    )
                    let sourceCol = try sourceFlatIndex(
                        retainedIndices: retainedColIndices,
                        tracedIndices: tracedIndices,
                        source: source,
                        selection: selection
                    )
                    return partial + source.operatorValue.entries[sourceRow, sourceCol]
                }
            }
        }
        return result
    }

    static func sourceFlatIndex(
        retainedIndices: [Int],
        tracedIndices: [Int],
        source: TensorProductEndomorphism,
        selection: ResolvedPartialTraceSelection
    ) throws -> Int {
        var fullIndices = Array(repeating: 0, count: source.factorDimensions.count)
        for (index, offset) in selection.retainedOffsets.enumerated() {
            fullIndices[offset.rawValue] = retainedIndices[index]
        }
        for (index, offset) in selection.tracedOffsets.enumerated() {
            fullIndices[offset.rawValue] = tracedIndices[index]
        }
        return try TensorIndexing.flatten(indices: fullIndices, factorDimensions: source.factorDimensions)
    }

    static func tensorIndexAssignments(for dimensions: [Int]) throws -> [[Int]] {
        let totalDimension = dimensions.reduce(1, *)
        return try (0..<totalDimension).map {
            try TensorIndexing.unflatten(index: $0, factorDimensions: dimensions)
        }
    }

    static func normalizedStateKet(
        from value: QuantumValue,
        config: QuantumMathConfig
    ) throws -> Ket {
        switch value {
        case let .ket(ket):
            return try normalize(ket, config: config)
        case let .bra(bra):
            return try normalize(dagger(bra), config: config)
        case .scalar, .oper:
            throw QuantumMathError.operationNotDefined("Density state creation requires bras or kets.")
        }
    }

    static func normalizedMixedDensityWeights(
        _ weights: [Scalar],
        config: QuantumMathConfig
    ) throws -> [Scalar] {
        let validatedWeights = try weights.map { weight -> Scalar in
            let value = weight.approximateValue
            guard abs(value.im) <= config.scalarComparisonEpsilon,
                  value.re >= -config.scalarComparisonEpsilon else {
                throw QuantumMathError.operationNotDefined(
                    "Mixed state weights must be real non-negative values."
                )
            }
            return value.re < 0 ? .zero : weight
        }
        let total = validatedWeights.reduce(.zero, +)
        guard !total.isZero(epsilon: config.scalarComparisonEpsilon) else {
            throw QuantumMathError.operationNotDefined("Mixed state weights must not all be zero.")
        }
        return try validatedWeights.map { weight in
            guard let normalized = weight.divided(by: total) else {
                throw QuantumMathError.operationNotDefined("Mixed state weights could not be normalized.")
            }
            return normalized
        }
    }

    static func normalizedCoefficients(
        _ coefficients: [Scalar],
        config: QuantumMathConfig
    ) throws -> [Scalar] {
        let normSquared = coefficients.reduce(0.0) { partial, scalar in
            partial + scalar.magnitudeSquaredApproximate
        }
        guard normSquared > config.normThreshold else {
            throw QuantumMathError.operationNotDefined("Zero vectors cannot be normalized.")
        }
        let norm = Scalar.approx(ComplexNumber(re: Foundation.sqrt(normSquared), im: 0))
        return try coefficients.map { coefficient in
            guard let normalized = coefficient.divided(by: norm) else {
                throw QuantumMathError.operationNotDefined("Zero vectors cannot be normalized.")
            }
            return normalized
        }
    }

    static func operatorDividing(_ operatorValue: Operator, by scalar: Scalar) throws -> Operator {
        let values = try operatorValue.entries.values.map { entry -> Scalar in
            guard let quotient = entry.divided(by: scalar) else {
                throw QuantumMathError.operationNotDefined(
                    "Operators with zero trace cannot be normalized to tr(A) = 1."
                )
            }
            return quotient
        }
        return try Operator(
            domain: operatorValue.domain,
            codomain: operatorValue.codomain,
            columnBasis: operatorValue.columnBasis,
            rowBasis: operatorValue.rowBasis,
            entries: try Matrix(
                rows: operatorValue.entries.rows,
                cols: operatorValue.entries.cols,
                values: values
            )
        )
    }

    static func densityOperatorSummary(
        for operatorValue: Operator,
        config: QuantumMathConfig
    ) -> DensityOperatorSummary? {
        guard operatorValue.domain == operatorValue.codomain,
              operatorValue.rowBasis == operatorValue.columnBasis,
              operatorValue.entries.isHermitian(tolerance: config.matrixTraitEpsilon),
              operatorValue.entries.isPositiveSemidefinite(tolerance: config.matrixTraitEpsilon),
              let trace = operatorValue.entries.trace(),
              trace.approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon) else {
            return nil
        }

        let traits = deriveTraits(from: operatorValue.entries, config: config)
        let recoveredKet = recoveredPureDensityKet(
            for: operatorValue,
            traits: traits,
            config: config
        )
        let purity: DensityPurity = recoveredKet == nil ? .mixed : .pure
        return DensityOperatorSummary(
            purity: purity,
            recoveredKet: recoveredKet,
            qubitBlochVector: qubitBlochVector(for: operatorValue)
        )
    }

    static func recoveredPureDensityKet(
        for operatorValue: Operator,
        traits: Set<OperatorTrait>,
        config: QuantumMathConfig
    ) -> Ket? {
        guard traits.contains(.projector),
              let pivotIndex = diagonalPivotIndex(for: operatorValue.entries),
              let pivotAmplitude = pivotAmplitude(
                from: operatorValue.entries[pivotIndex, pivotIndex],
                config: config
              ),
              !pivotAmplitude.isZero(epsilon: config.scalarComparisonEpsilon) else {
            return nil
        }

        let coefficients = (0..<operatorValue.entries.rows).compactMap { row in
            operatorValue.entries[row, pivotIndex].divided(by: pivotAmplitude)
        }
        guard coefficients.count == operatorValue.entries.rows,
              let ket = try? Ket(
                space: operatorValue.domain,
                basis: operatorValue.columnBasis,
                coefficients: coefficients
              ),
              let reconstructed = try? outerProduct(ket, dagger(ket)),
              reconstructed.entries.approximatelyEquals(
                operatorValue.entries,
                epsilon: config.matrixTraitEpsilon
              ) else {
            return nil
        }
        return ket
    }

    static func diagonalPivotIndex(for entries: Matrix<Scalar>) -> Int? {
        (0..<entries.rows).max {
            entries[$0, $0].magnitudeSquaredApproximate < entries[$1, $1].magnitudeSquaredApproximate
        }
    }

    static func pivotAmplitude(from diagonal: Scalar, config: QuantumMathConfig) -> Scalar? {
        let approximate = diagonal.approximateValue
        guard abs(approximate.im) <= config.matrixTraitEpsilon,
              approximate.re >= -config.matrixTraitEpsilon else {
            return nil
        }
        return .approx(ComplexNumber(re: Foundation.sqrt(max(approximate.re, 0)), im: 0))
    }

    static func qubitBlochVector(for operatorValue: Operator) -> QubitBlochVector? {
        guard operatorValue.domain == .atomic(.qubit),
              operatorValue.columnBasis == .computational(for: operatorValue.domain) else {
            return nil
        }

        let offDiagonal = operatorValue.entries[0, 1]
        return QubitBlochVector(
            x: offDiagonal + offDiagonal.conjugated,
            y: Scalar.i * (offDiagonal - offDiagonal.conjugated),
            z: operatorValue.entries[0, 0] - operatorValue.entries[1, 1]
        )
    }

    static func qubitBlochVector(forNormalizedCoefficients coefficients: [Scalar]) -> QubitBlochVector? {
        guard coefficients.count == 2 else {
            return nil
        }

        let alpha = coefficients[0].approximateValue
        let beta = coefficients[1].approximateValue
        let alphaTimesBetaConjugate = alpha * beta.conjugated
        let x = Scalar.approx(ComplexNumber(re: 2 * alphaTimesBetaConjugate.re, im: 0))
        let y = Scalar.approx(ComplexNumber(re: 2 * alphaTimesBetaConjugate.im, im: 0))
        let z = Scalar.approx(
            ComplexNumber(
                re: alpha.magnitudeSquared - beta.magnitudeSquared,
                im: 0
            )
        )
        return QubitBlochVector(x: x, y: y, z: z)
    }

    static func realFiniteComponent(_ scalar: Scalar, epsilon: Double) -> Double? {
        let value = scalar.approximateValue
        guard abs(value.im) <= epsilon, value.re.isFinite else {
            return nil
        }
        return value.re
    }

    static func requireSameSpace(_ lhs: Space, _ rhs: Space) throws {
        guard lhs == rhs else {
            throw QuantumMathError.incompatibleSpaces(expected: lhs, actual: rhs)
        }
    }

    static func requireSameBasis(_ lhs: Basis, _ rhs: Basis) throws {
        guard lhs == rhs else {
            throw QuantumMathError.incompatibleBases(lhs: lhs, rhs: rhs)
        }
    }
}
