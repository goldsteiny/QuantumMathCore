import Foundation

public struct QuantumReferenceID: Hashable, Comparable, Sendable, Codable {
    public let rawValue: String

    public init(_ rawValue: String) {
        self.rawValue = rawValue
    }

    public static func < (lhs: QuantumReferenceID, rhs: QuantumReferenceID) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public protocol ValueResolver {
    func resolve(_ reference: QuantumReferenceID) throws -> QuantumValue
}

public enum QuantumUnaryOperation: String, Hashable, Sendable, Codable, CaseIterable {
    case dagger
    case normalize
    case trace
    case transpose
}

public enum QuantumBinaryOperation: String, Hashable, Sendable, Codable, CaseIterable {
    case innerProduct
    case outerProduct
    case applyOperator
    case applyBraAfterOperator
    case tensor
    case scale
    case add
    case subtract
    case compose
    case commutator
    case anticommutator
}

public enum QuantumOrderedOperation: String, Hashable, Sendable, Codable, CaseIterable {
    case tensor
    case compose
    case sandwich
}

public struct QuantumWeightedSumTerm: Hashable, Sendable, Codable {
    public let reference: QuantumReferenceID
    public let weight: Scalar

    public init(reference: QuantumReferenceID, weight: Scalar) {
        self.reference = reference
        self.weight = weight
    }
}

public struct QuantumWeightedSumExpression: Hashable, Sendable, Codable {
    public let terms: [QuantumWeightedSumTerm]
    public let normalization: WeightedSumNormalization

    public init(
        terms: [QuantumWeightedSumTerm],
        normalization: WeightedSumNormalization
    ) {
        self.terms = terms
        self.normalization = normalization
    }
}

public enum QubitBlochStateExpression: Hashable, Sendable, Codable {
    case density(QubitBlochVector)
    case ket(QubitBlochAngles)
    case bra(QubitBlochAngles)
}

public enum QuantumExpression: Hashable, Sendable, Codable {
    case literal(QuantumValue)
    case reference(QuantumReferenceID)
    case unary(QuantumUnaryOperation, QuantumReferenceID)
    case binary(QuantumBinaryOperation, QuantumReferenceID, QuantumReferenceID)
    case operatorPower(reference: QuantumReferenceID, exponent: Int)
    case ordered(QuantumOrderedOperation, [QuantumReferenceID])
    case weightedSum(QuantumWeightedSumExpression)
    case pureDensityState(QuantumReferenceID)
    case mixedDensityState([QuantumWeightedSumTerm])
    case partialTrace(reference: QuantumReferenceID, selection: PartialTraceSelection)
    case explicitBasisConversion(reference: QuantumReferenceID, targetBasis: Basis)
    case blochState(QubitBlochStateExpression)

    public var dependencyIDs: [QuantumReferenceID] {
        switch self {
        case .literal:
            return []
        case let .reference(reference):
            return [reference]
        case let .unary(_, reference):
            return [reference]
        case let .binary(_, lhs, rhs):
            return [lhs, rhs]
        case let .operatorPower(reference, _):
            return [reference]
        case let .ordered(_, references):
            return references
        case let .weightedSum(expression):
            return expression.terms.map(\.reference)
        case let .pureDensityState(reference):
            return [reference]
        case let .mixedDensityState(terms):
            return terms.map(\.reference)
        case let .partialTrace(reference, _):
            return [reference]
        case let .explicitBasisConversion(reference, _):
            return [reference]
        case .blochState:
            return []
        }
    }
}

public struct QuantumOperation {
    public let config: QuantumMathConfig

    public init(config: QuantumMathConfig = .ketStepsDefault) {
        self.config = config
    }

    public func evaluate(
        expression: QuantumExpression,
        resolver: ValueResolver
    ) throws -> QuantumValue {
        switch expression {
        case let .literal(value):
            return value
        case let .reference(reference):
            return try resolver.resolve(reference)
        case let .unary(operation, reference):
            let operand = try resolver.resolve(reference)
            return try evaluateUnary(operation, operand: operand)
        case let .binary(operation, lhs, rhs):
            return try evaluateBinary(
                operation,
                lhs: resolver.resolve(lhs),
                rhs: resolver.resolve(rhs)
            )
        case let .operatorPower(reference, exponent):
            guard case let .oper(operatorValue) = try resolver.resolve(reference) else {
                throw QuantumMathError.operationNotDefined("Power requires an operator.")
            }
            return .oper(try QuantumDomain.power(operatorValue, exponent: exponent))
        case let .ordered(operation, references):
            let operands = try references.map { try resolver.resolve($0) }
            return try evaluateOrdered(operation, operands: operands)
        case let .weightedSum(expression):
            let terms = try expression.terms.map {
                (weight: $0.weight, value: try resolver.resolve($0.reference))
            }
            return try QuantumDomain.weightedSum(
                terms: terms,
                normalization: expression.normalization,
                config: config
            )
        case let .pureDensityState(reference):
            return .oper(try QuantumDomain.pureDensityState(
                from: resolver.resolve(reference),
                config: config
            ))
        case let .mixedDensityState(terms):
            return .oper(try QuantumDomain.mixedDensityState(
                terms: try terms.map {
                    (weight: $0.weight, value: try resolver.resolve($0.reference))
                },
                config: config
            ))
        case let .partialTrace(reference, selection):
            guard case let .oper(operatorValue) = try resolver.resolve(reference) else {
                throw QuantumMathError.operationNotDefined("Partial trace requires an operator.")
            }
            return .oper(try QuantumDomain.partialTrace(operatorValue, tracing: selection))
        case let .explicitBasisConversion(reference, targetBasis):
            return try QuantumDomain.convert(resolver.resolve(reference), to: targetBasis)
        case let .blochState(stateExpression):
            switch stateExpression {
            case let .density(vector):
                return .oper(try QuantumDomain.densityOperator(
                    fromBlochVector: vector,
                    config: config
                ))
            case let .ket(angles):
                return .ket(try QuantumDomain.ket(
                    fromBlochAngles: angles,
                    config: config
                ))
            case let .bra(angles):
                return .bra(try QuantumDomain.bra(
                    fromBlochAngles: angles,
                    config: config
                ))
            }
        }
    }

    private func evaluateUnary(
        _ operation: QuantumUnaryOperation,
        operand: QuantumValue
    ) throws -> QuantumValue {
        switch operation {
        case .dagger:
            return try QuantumDomain.dagger(operand)
        case .normalize:
            switch operand {
            case let .ket(ket):
                return .ket(try QuantumDomain.normalize(ket, config: config))
            case let .bra(bra):
                return .bra(try QuantumDomain.normalize(bra, config: config))
            case let .oper(operatorValue):
                return .oper(try QuantumDomain.normalize(operatorValue))
            case .scalar:
                throw QuantumMathError.operationNotDefined(
                    "Normalize is only defined for bras, kets, and operators."
                )
            }
        case .trace:
            guard case let .oper(operatorValue) = operand else {
                throw QuantumMathError.operationNotDefined("Trace requires an operator.")
            }
            return .scalar(try QuantumDomain.trace(operatorValue))
        case .transpose:
            guard case let .oper(operatorValue) = operand else {
                throw QuantumMathError.operationNotDefined("Transpose requires an operator.")
            }
            return .oper(try QuantumDomain.transpose(operatorValue))
        }
    }

    private func evaluateBinary(
        _ operation: QuantumBinaryOperation,
        lhs: QuantumValue,
        rhs: QuantumValue
    ) throws -> QuantumValue {
        switch operation {
        case .innerProduct:
            guard case let .bra(bra) = lhs, case let .ket(ket) = rhs else {
                throw QuantumMathError.operationNotDefined("Inner product requires a bra and a ket.")
            }
            return .scalar(try QuantumDomain.innerProduct(bra, ket))
        case .outerProduct:
            guard case let .ket(ket) = lhs, case let .bra(bra) = rhs else {
                throw QuantumMathError.operationNotDefined("Outer product requires a ket and a bra.")
            }
            return .oper(try QuantumDomain.outerProduct(ket, bra))
        case .applyOperator:
            guard case let .oper(operatorValue) = lhs, case let .ket(ket) = rhs else {
                throw QuantumMathError.operationNotDefined(
                    "Operator application requires an operator and a ket."
                )
            }
            return .ket(try QuantumDomain.applyOperator(operatorValue, ket))
        case .applyBraAfterOperator:
            guard case let .bra(bra) = lhs, case let .oper(operatorValue) = rhs else {
                throw QuantumMathError.operationNotDefined(
                    "Bra-after-operator requires a bra and an operator."
                )
            }
            return .bra(try QuantumDomain.applyBraAfterOperator(bra, operatorValue))
        case .tensor:
            switch (lhs, rhs) {
            case let (.ket(left), .ket(right)):
                return .ket(try QuantumDomain.tensor(left, right, config: config))
            case let (.bra(left), .bra(right)):
                return .bra(try QuantumDomain.tensor(left, right, config: config))
            case let (.oper(left), .oper(right)):
                return .oper(try QuantumDomain.tensor(left, right, config: config))
            default:
                throw QuantumMathError.operationNotDefined(
                    "Tensor product requires matching kets, bras, or square operators."
                )
            }
        case .scale:
            if case let .scalar(scalar) = lhs {
                return try QuantumDomain.scale(scalar, value: rhs)
            }
            if case let .scalar(scalar) = rhs {
                return try QuantumDomain.scale(scalar, value: lhs)
            }
            throw QuantumMathError.operationNotDefined("Scale requires one scalar operand.")
        case .add:
            return try QuantumDomain.add(lhs, rhs)
        case .subtract:
            return try QuantumDomain.subtract(lhs, rhs)
        case .compose:
            guard case let .oper(left) = lhs, case let .oper(right) = rhs else {
                throw QuantumMathError.operationNotDefined("Composition requires two operators.")
            }
            return .oper(try QuantumDomain.compose(left, right))
        case .commutator:
            guard case let .oper(left) = lhs, case let .oper(right) = rhs else {
                throw QuantumMathError.operationNotDefined("Commutator requires two operators.")
            }
            return .oper(try QuantumDomain.commutator(left, right))
        case .anticommutator:
            guard case let .oper(left) = lhs, case let .oper(right) = rhs else {
                throw QuantumMathError.operationNotDefined("Anti-commutator requires two operators.")
            }
            return .oper(try QuantumDomain.anticommutator(left, right))
        }
    }

    private func evaluateOrdered(
        _ operation: QuantumOrderedOperation,
        operands: [QuantumValue]
    ) throws -> QuantumValue {
        guard operands.count >= 2 else {
            throw QuantumMathError.operationNotDefined("Ordered operations require at least two operands.")
        }
        switch operation {
        case .tensor:
            return try operands.dropFirst().reduce(operands[0]) { partial, next in
                try evaluateBinary(.tensor, lhs: partial, rhs: next)
            }
        case .compose:
            return try operands.dropFirst().reduce(operands[0]) { partial, next in
                try evaluateBinary(.compose, lhs: partial, rhs: next)
            }
        case .sandwich:
            guard operands.count >= 3,
                  case let .bra(bra)? = operands.first,
                  case let .ket(ket)? = operands.last else {
                throw QuantumMathError.operationNotDefined(
                    "Sandwich requires a bra first, operators in the middle, and a ket last."
                )
            }
            let operators = operands.dropFirst().dropLast()
            guard let firstOperatorValue = operators.first,
                  case let .oper(firstOperator) = firstOperatorValue else {
                throw QuantumMathError.operationNotDefined("Sandwich requires operators between the bra and ket.")
            }
            let chain = try operators.dropFirst().reduce(firstOperator) { partial, next in
                guard case let .oper(nextOperator) = next else {
                    throw QuantumMathError.operationNotDefined("Sandwich requires operators between the bra and ket.")
                }
                return try QuantumDomain.compose(partial, nextOperator)
            }
            return .scalar(try QuantumDomain.sandwich(bra, chain, ket))
        }
    }
}
