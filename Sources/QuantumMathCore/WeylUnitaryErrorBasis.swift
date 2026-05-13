import Foundation
import ExactValueRecoveryCore

public struct WeylIndex: Hashable, Comparable, Sendable, Codable {
    public let dimension: Int
    public let shift: Int
    public let phase: Int

    public init(dimension: Int, shift: Int, phase: Int) throws {
        guard dimension >= 2 else {
            throw QuantumMathError.operationNotDefined("Weyl bases require dimension >= 2.")
        }
        guard shift >= 0, shift < dimension, phase >= 0, phase < dimension else {
            throw QuantumMathError.operationNotDefined(
                "Weyl shift and phase indices must be in 0..<dimension."
            )
        }
        self.dimension = dimension
        self.shift = shift
        self.phase = phase
    }

    public init(dimension: Int, flatIndex: Int) throws {
        guard dimension >= 2 else {
            throw QuantumMathError.operationNotDefined("Weyl bases require dimension >= 2.")
        }
        guard flatIndex >= 0, flatIndex < dimension * dimension else {
            throw QuantumMathError.operationNotDefined(
                "Weyl flat index must be in 0..<dimension^2."
            )
        }
        try self.init(
            dimension: dimension,
            shift: flatIndex % dimension,
            phase: flatIndex / dimension
        )
    }

    public var flatIndex: Int {
        (phase * dimension) + shift
    }

    public static func < (lhs: WeylIndex, rhs: WeylIndex) -> Bool {
        if lhs.dimension != rhs.dimension {
            return lhs.dimension < rhs.dimension
        }
        return lhs.flatIndex < rhs.flatIndex
    }
}

public struct UnitaryErrorBasisElement: Hashable, Sendable, Codable {
    public let index: WeylIndex
    public let operatorValue: Operator

    public init(index: WeylIndex, operatorValue: Operator) {
        self.index = index
        self.operatorValue = operatorValue
    }
}

public struct WeylUnitaryErrorBasis: Hashable, Sendable, Codable {
    public let dimension: Int
    public let basis: Basis
    public let elements: [UnitaryErrorBasisElement]

    public init(
        dimension: Int,
        basis: Basis,
        elements: [UnitaryErrorBasisElement],
        config: QuantumMathConfig = .ketStepsDefault
    ) throws {
        guard dimension >= 2 else {
            throw QuantumMathError.operationNotDefined("Weyl bases require dimension >= 2.")
        }
        guard basis.space.factors.count == 1, basis.space.dimension == dimension else {
            throw QuantumMathError.operationNotDefined(
                "A Weyl basis must use a one-factor space matching its dimension."
            )
        }
        guard elements.count == dimension * dimension else {
            throw QuantumMathError.operationNotDefined(
                "A dimension-\(dimension) unitary error basis must contain \(dimension * dimension) elements."
            )
        }

        let sortedElements = elements.sorted { $0.index < $1.index }
        guard Set(sortedElements.map(\.index)).count == sortedElements.count else {
            throw QuantumMathError.operationNotDefined("Weyl basis indices must be unique.")
        }

        for (expectedFlatIndex, element) in sortedElements.enumerated() {
            guard element.index.dimension == dimension,
                  element.index.flatIndex == expectedFlatIndex else {
                throw QuantumMathError.operationNotDefined(
                    "Weyl basis elements must cover every shift/phase index exactly once."
                )
            }
            try Self.validate(
                element.operatorValue,
                matches: element.index,
                basis: basis,
                config: config
            )
        }

        self.dimension = dimension
        self.basis = basis
        self.elements = sortedElements
    }

    public func element(for index: WeylIndex) throws -> UnitaryErrorBasisElement {
        guard index.dimension == dimension,
              let element = elements.first(where: { $0.index == index }) else {
            throw QuantumMathError.operationNotDefined("Weyl basis element is not present.")
        }
        return element
    }

    public func element(flatIndex: Int) throws -> UnitaryErrorBasisElement {
        try element(for: WeylIndex(dimension: dimension, flatIndex: flatIndex))
    }

    public func hilbertSchmidtInnerProduct(
        _ lhs: WeylIndex,
        _ rhs: WeylIndex
    ) throws -> Scalar {
        try QuantumDomain.hilbertSchmidtInnerProduct(
            element(for: lhs).operatorValue,
            element(for: rhs).operatorValue
        )
    }

    public func validateHilbertSchmidtOrthogonality(
        config: QuantumMathConfig = .ketStepsDefault
    ) throws {
        for lhs in elements {
            for rhs in elements {
                let innerProduct = try QuantumDomain.hilbertSchmidtInnerProduct(
                    lhs.operatorValue,
                    rhs.operatorValue
                )
                let expected: Scalar = lhs.index == rhs.index
                    ? Scalar(real: Rational(dimension))
                    : .zero
                guard innerProduct.approximatelyEquals(
                    expected,
                    epsilon: config.matrixTraitEpsilon
                ) else {
                    throw QuantumMathError.operationNotDefined(
                        "Weyl basis failed Hilbert-Schmidt orthogonality validation."
                    )
                }
            }
        }
    }

    private static func validate(
        _ operatorValue: Operator,
        matches index: WeylIndex,
        basis: Basis,
        config: QuantumMathConfig
    ) throws {
        guard operatorValue.domain == basis.space,
              operatorValue.codomain == basis.space,
              operatorValue.columnBasis == basis,
              operatorValue.rowBasis == basis else {
            throw QuantumMathError.operationNotDefined(
                "Weyl basis operators must be endomorphisms on the declared basis."
            )
        }
        guard operatorValue.entries.isUnitary(tolerance: config.matrixTraitEpsilon) else {
            throw QuantumMathError.operationNotDefined("Weyl basis operators must be unitary.")
        }

        let expected = try QuantumDomain.weylOperator(
            index: index,
            basis: basis
        )
        guard operatorValue.entries.approximatelyEquals(
            expected.entries,
            epsilon: config.matrixTraitEpsilon
        ) else {
            throw QuantumMathError.operationNotDefined(
                "Weyl basis operator entries do not match X^shift Z^phase."
            )
        }
    }
}

public struct WeylBellBasisElement: Hashable, Sendable, Codable {
    public let index: WeylIndex
    public let state: Ket
    public let projector: Operator

    public init(index: WeylIndex, state: Ket, projector: Operator) {
        self.index = index
        self.state = state
        self.projector = projector
    }
}

public struct WeylBellBasis: Hashable, Sendable, Codable {
    public let dimension: Int
    public let basis: Basis
    public let elements: [WeylBellBasisElement]

    public init(
        dimension: Int,
        basis: Basis,
        elements: [WeylBellBasisElement]
    ) throws {
        guard dimension >= 2 else {
            throw QuantumMathError.operationNotDefined("Weyl Bell bases require dimension >= 2.")
        }
        guard basis.space.factors.map(\.dimension) == [dimension, dimension] else {
            throw QuantumMathError.operationNotDefined(
                "A Weyl Bell basis must live on two equal-dimensional factors."
            )
        }
        guard elements.count == dimension * dimension else {
            throw QuantumMathError.operationNotDefined(
                "A dimension-\(dimension) Weyl Bell basis must contain \(dimension * dimension) elements."
            )
        }

        let sortedElements = elements.sorted { $0.index < $1.index }
        guard Set(sortedElements.map(\.index)).count == sortedElements.count else {
            throw QuantumMathError.operationNotDefined("Weyl Bell basis indices must be unique.")
        }

        for (expectedFlatIndex, element) in sortedElements.enumerated() {
            guard element.index.dimension == dimension,
                  element.index.flatIndex == expectedFlatIndex,
                  element.state.space == basis.space,
                  element.state.basis == basis,
                  element.projector.domain == basis.space,
                  element.projector.codomain == basis.space,
                  element.projector.columnBasis == basis,
                  element.projector.rowBasis == basis else {
                throw QuantumMathError.operationNotDefined(
                    "Weyl Bell basis elements must match the declared two-factor basis."
                )
            }
        }

        self.dimension = dimension
        self.basis = basis
        self.elements = sortedElements
    }

    public func element(for index: WeylIndex) throws -> WeylBellBasisElement {
        guard index.dimension == dimension,
              let element = elements.first(where: { $0.index == index }) else {
            throw QuantumMathError.operationNotDefined("Weyl Bell basis element is not present.")
        }
        return element
    }
}

public enum CommunicationFactorRole: String, Hashable, Sendable, Codable {
    case input
    case aliceShare
    case bobShare
}

public struct DenseCodingScheme: Hashable, Sendable, Codable {
    public let unitaryErrorBasis: WeylUnitaryErrorBasis
    public let factorOrder: [CommunicationFactorRole]
    public let encodedIndex: WeylIndex
    public let resourceState: Ket
    public let aliceEncodingOperation: Operator
    public let liftedEncodingOperation: Operator
    public let encodedJointState: Ket
    public let measurementBasis: WeylBellBasis
    public let decodedIndex: WeylIndex

    public init(
        unitaryErrorBasis: WeylUnitaryErrorBasis,
        factorOrder: [CommunicationFactorRole] = [.aliceShare, .bobShare],
        encodedIndex: WeylIndex,
        resourceState: Ket,
        aliceEncodingOperation: Operator,
        liftedEncodingOperation: Operator,
        encodedJointState: Ket,
        measurementBasis: WeylBellBasis,
        decodedIndex: WeylIndex
    ) {
        self.unitaryErrorBasis = unitaryErrorBasis
        self.factorOrder = factorOrder
        self.encodedIndex = encodedIndex
        self.resourceState = resourceState
        self.aliceEncodingOperation = aliceEncodingOperation
        self.liftedEncodingOperation = liftedEncodingOperation
        self.encodedJointState = encodedJointState
        self.measurementBasis = measurementBasis
        self.decodedIndex = decodedIndex
    }
}

public struct TeleportationOutcome: Hashable, Sendable, Codable {
    public let index: WeylIndex
    public let measurementState: Ket
    public let measurementProjector: Operator
    public let probability: Scalar
    public let bobStateBeforeCorrection: Ket
    public let correctionOperation: Operator
    public let bobStateAfterCorrection: Ket
    public let fullStateAfterMeasurement: Ket
    public let fullStateAfterCorrection: Ket

    public init(
        index: WeylIndex,
        measurementState: Ket,
        measurementProjector: Operator,
        probability: Scalar,
        bobStateBeforeCorrection: Ket,
        correctionOperation: Operator,
        bobStateAfterCorrection: Ket,
        fullStateAfterMeasurement: Ket,
        fullStateAfterCorrection: Ket
    ) {
        self.index = index
        self.measurementState = measurementState
        self.measurementProjector = measurementProjector
        self.probability = probability
        self.bobStateBeforeCorrection = bobStateBeforeCorrection
        self.correctionOperation = correctionOperation
        self.bobStateAfterCorrection = bobStateAfterCorrection
        self.fullStateAfterMeasurement = fullStateAfterMeasurement
        self.fullStateAfterCorrection = fullStateAfterCorrection
    }
}

public struct TeleportationScheme: Hashable, Sendable, Codable {
    public let dimension: Int
    public let factorOrder: [CommunicationFactorRole]
    public let inputState: Ket
    public let resourceState: Ket
    public let initialState: Ket
    public let measurementBasis: WeylBellBasis
    public let outcomes: [TeleportationOutcome]

    public init(
        dimension: Int,
        factorOrder: [CommunicationFactorRole] = [.input, .aliceShare, .bobShare],
        inputState: Ket,
        resourceState: Ket,
        initialState: Ket,
        measurementBasis: WeylBellBasis,
        outcomes: [TeleportationOutcome]
    ) {
        self.dimension = dimension
        self.factorOrder = factorOrder
        self.inputState = inputState
        self.resourceState = resourceState
        self.initialState = initialState
        self.measurementBasis = measurementBasis
        self.outcomes = outcomes
    }
}

public extension QuantumDomain {
    static func rootOfUnity(order: Int, power: Int) throws -> Scalar {
        guard order >= 1 else {
            throw QuantumMathError.operationNotDefined("Root-of-unity order must be positive.")
        }
        return .exact(.canonicalRootOfUnity(order: order, power: power))
    }

    static func inverseSquareRoot(_ dimension: Int) throws -> Scalar {
        guard dimension >= 1 else {
            throw QuantumMathError.operationNotDefined("Inverse square roots require positive dimension.")
        }
        return .exact(
            .canonicalSignedRationalTimesSqrt(
                coefficient: Rational(1, dimension),
                radicand: dimension
            )
        )
    }

    static func identityOperator(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        let basis = try computationalBasis(dimension: dimension, config: config)
        return try Operator(
            domain: basis.space,
            codomain: basis.space,
            columnBasis: basis,
            rowBasis: basis,
            entries: .identity(size: dimension)
        )
    }

    static func weylShiftOperator(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        try weylOperator(
            index: WeylIndex(dimension: dimension, shift: 1, phase: 0),
            basis: computationalBasis(dimension: dimension, config: config)
        )
    }

    static func weylPhaseOperator(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        try weylOperator(
            index: WeylIndex(dimension: dimension, shift: 0, phase: 1),
            basis: computationalBasis(dimension: dimension, config: config)
        )
    }

    static func weylOperator(
        dimension: Int,
        shift: Int,
        phase: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        try weylOperator(
            index: WeylIndex(dimension: dimension, shift: shift, phase: phase),
            basis: computationalBasis(dimension: dimension, config: config)
        )
    }

    static func weylOperator(
        dimension: Int,
        flatIndex: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        try weylOperator(
            index: WeylIndex(dimension: dimension, flatIndex: flatIndex),
            basis: computationalBasis(dimension: dimension, config: config)
        )
    }

    static func weylOperator(index: WeylIndex, basis: Basis) throws -> Operator {
        guard basis.space.factors.count == 1,
              basis.space.dimension == index.dimension else {
            throw QuantumMathError.operationNotDefined(
                "Weyl operators require a matching one-factor basis."
            )
        }

        let dimension = index.dimension
        let values = try (0..<dimension).flatMap { row in
            try (0..<dimension).map { col in
                guard row == positiveModulo(col + index.shift, dimension) else {
                    return Scalar.zero
                }
                return try rootOfUnity(order: dimension, power: index.phase * col)
            }
        }
        let entries = try Matrix(
            rows: dimension,
            cols: dimension,
            values: values
        )

        return try Operator(
            domain: basis.space,
            codomain: basis.space,
            columnBasis: basis,
            rowBasis: basis,
            entries: entries
        )
    }

    static func weylUnitaryErrorBasis(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> WeylUnitaryErrorBasis {
        try requireDimension(dimension, within: config, purpose: "Weyl unitary error basis")
        let basis = try computationalBasis(dimension: dimension, config: config)
        let elements = try (0..<dimension).flatMap { phase in
            try (0..<dimension).map { shift in
                let index = try WeylIndex(dimension: dimension, shift: shift, phase: phase)
                return UnitaryErrorBasisElement(
                    index: index,
                    operatorValue: try weylOperator(index: index, basis: basis)
                )
            }
        }
        return try WeylUnitaryErrorBasis(
            dimension: dimension,
            basis: basis,
            elements: elements,
            config: config
        )
    }

    static func hilbertSchmidtInnerProduct(_ lhs: Operator, _ rhs: Operator) throws -> Scalar {
        guard lhs.domain == rhs.domain,
              lhs.codomain == rhs.codomain,
              lhs.columnBasis == rhs.columnBasis,
              lhs.rowBasis == rhs.rowBasis else {
            throw QuantumMathError.operationNotDefined(
                "Hilbert-Schmidt inner product requires operators on matching spaces and bases."
            )
        }
        return zip(lhs.entries.values, rhs.entries.values).reduce(.zero) { partial, pair in
            partial + (pair.0.conjugated * pair.1)
        }
    }

    static func liftOneFactorOperator(
        _ operatorValue: Operator,
        toRawFactorOffset rawOffset: Int,
        in targetSpace: Space,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        try liftOneFactorOperator(
            operatorValue,
            toFactorOffset: TensorFactorOffset(rawValue: rawOffset),
            in: targetSpace,
            config: config
        )
    }

    static func liftOneFactorOperator(
        _ operatorValue: Operator,
        toFactorOffset factorOffset: TensorFactorOffset,
        in targetSpace: Space,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Operator {
        guard operatorValue.domain == operatorValue.codomain,
              operatorValue.domain.factors.count == 1,
              operatorValue.columnBasis == operatorValue.rowBasis else {
            throw QuantumMathError.operationNotDefined(
                "Only square one-factor operators can be lifted into tensor spaces."
            )
        }

        let factors = targetSpace.factors
        guard factorOffset.rawValue < factors.count else {
            throw QuantumMathError.operationNotDefined("Target tensor factor is out of range.")
        }
        guard factors[factorOffset.rawValue].dimension == operatorValue.domain.dimension else {
            throw QuantumMathError.operationNotDefined(
                "Lifted operator dimension must match the target tensor factor."
            )
        }

        let operands = try factors.indices.map { offset -> Operator in
            if offset == factorOffset.rawValue {
                return operatorValue
            }
            return try identityOperator(dimension: factors[offset].dimension, config: config)
        }
        return try tensorProduct(operators: operands, config: config)
    }

    static func maximallyEntangledState(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> Ket {
        try requireJointDimension(
            dimension,
            factorCount: 2,
            within: config,
            purpose: "Maximally entangled state"
        )

        let space = try repeatedQuditSpace(
            dimension: dimension,
            factorCount: 2,
            config: config
        )
        let basis = Basis.computational(for: space)
        let normalization = try inverseSquareRoot(dimension)
        var coefficients = Array(repeating: Scalar.zero, count: space.dimension)

        for index in 0..<dimension {
            let flat = try TensorIndexing.flatten(
                indices: [index, index],
                factorDimensions: [dimension, dimension]
            )
            coefficients[flat] = normalization
        }

        return try Ket(space: space, basis: basis, coefficients: coefficients)
    }

    static func weylBellBasis(
        dimension: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> WeylBellBasis {
        try requireJointDimension(
            dimension,
            factorCount: 2,
            within: config,
            purpose: "Weyl Bell basis"
        )

        let unitaryBasis = try weylUnitaryErrorBasis(dimension: dimension, config: config)
        let resource = try maximallyEntangledState(dimension: dimension, config: config)
        let elements = try unitaryBasis.elements.map { element -> WeylBellBasisElement in
            let lifted = try liftOneFactorOperator(
                element.operatorValue,
                toRawFactorOffset: 0,
                in: resource.space,
                config: config
            )
            let state = try applyOperator(lifted, resource)
            let projector = try outerProduct(state, dagger(state))
            return WeylBellBasisElement(
                index: element.index,
                state: state,
                projector: projector
            )
        }
        return try WeylBellBasis(
            dimension: dimension,
            basis: resource.basis,
            elements: elements
        )
    }

    static func denseCodingScheme(
        dimension: Int,
        encodedIndex: WeylIndex,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> DenseCodingScheme {
        try requireJointDimension(
            dimension,
            factorCount: 2,
            within: config,
            purpose: "Dense coding"
        )
        guard encodedIndex.dimension == dimension else {
            throw QuantumMathError.operationNotDefined(
                "Dense coding encoded index dimension must match the scheme dimension."
            )
        }

        let unitaryBasis = try weylUnitaryErrorBasis(dimension: dimension, config: config)
        let encodingElement = try unitaryBasis.element(for: encodedIndex)
        let resource = try maximallyEntangledState(dimension: dimension, config: config)
        let liftedEncoding = try liftOneFactorOperator(
            encodingElement.operatorValue,
            toRawFactorOffset: 0,
            in: resource.space,
            config: config
        )
        let encodedState = try applyOperator(liftedEncoding, resource)
        let measurementBasis = try weylBellBasis(dimension: dimension, config: config)
        guard let decodedIndex = matchingBellIndex(
            for: encodedState,
            in: measurementBasis,
            config: config
        ) else {
            throw QuantumMathError.operationNotDefined("Dense coding state could not be decoded.")
        }

        return DenseCodingScheme(
            unitaryErrorBasis: unitaryBasis,
            encodedIndex: encodedIndex,
            resourceState: resource,
            aliceEncodingOperation: encodingElement.operatorValue,
            liftedEncodingOperation: liftedEncoding,
            encodedJointState: encodedState,
            measurementBasis: measurementBasis,
            decodedIndex: decodedIndex
        )
    }

    static func denseCodingScheme(
        dimension: Int,
        flatIndex: Int,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> DenseCodingScheme {
        try denseCodingScheme(
            dimension: dimension,
            encodedIndex: WeylIndex(dimension: dimension, flatIndex: flatIndex),
            config: config
        )
    }

    static func denseCommunicationTranscript(
        resource: MaximallyEntangledResource,
        message: WeylMessage,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> DenseCommunicationTranscript {
        guard message.dimension == resource.dimension else {
            throw QuantumMathError.operationNotDefined(
                "Dense communication message dimension must match the resource dimension."
            )
        }

        let unitaryBasis = try weylUnitaryErrorBasis(dimension: resource.dimension, config: config)
        let encodingElement = try unitaryBasis.element(for: message.index)
        let liftedEncoding = try liftOneFactorOperator(
            encodingElement.operatorValue,
            toRawFactorOffset: 0,
            in: resource.state.space,
            config: config
        )
        let encodedState = try applyOperator(liftedEncoding, resource.state)
        let measurementBasis = try resourceRelativeWeylBellBasis(resource: resource, config: config)
        guard let decodedIndex = matchingBellIndex(
            for: encodedState,
            in: measurementBasis,
            config: config
        ) else {
            throw QuantumMathError.operationNotDefined("Dense communication state could not be decoded.")
        }

        let decodedMessage = WeylMessage(index: decodedIndex)
        let measurementElement = try measurementBasis.element(for: decodedIndex)
        let succeeded = decodedMessage == message

        return DenseCommunicationTranscript(
            resource: resource,
            message: message,
            aliceEncodingOperation: encodingElement.operatorValue,
            liftedEncodingOperation: liftedEncoding,
            encodedJointState: encodedState,
            measurementBasis: measurementBasis,
            bobMeasurementState: measurementElement.state,
            bobMeasurementProjector: measurementElement.projector,
            decodedMessage: decodedMessage,
            verification: ProtocolVerification(
                succeeded: succeeded,
                metric: succeeded ? .one : .zero,
                message: succeeded
                    ? "Decoded message matches Alice's message."
                    : "Decoded message does not match Alice's message."
            )
        )
    }

    static func teleportationScheme(
        inputState: Ket,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> TeleportationScheme {
        guard inputState.space.factors.count == 1,
              let dimension = inputState.space.factors.first?.dimension else {
            throw QuantumMathError.operationNotDefined(
                "Teleportation input must be a one-qudit ket."
            )
        }
        guard inputState.basis.isComputationalCoordinateBasis else {
            throw QuantumMathError.operationNotDefined(
                "Weyl teleportation currently requires computational-coordinate input."
            )
        }
        try requireNormalized(inputState, config: config, purpose: "Teleportation input")
        try requireJointDimension(
            dimension,
            factorCount: 3,
            within: config,
            purpose: "Teleportation"
        )

        let unitaryBasis = try weylUnitaryErrorBasis(dimension: dimension, config: config)
        let resource = try maximallyEntangledState(dimension: dimension, config: config)
        let initialState = try tensor(inputState, resource, config: config)
        let measurementBasis = try weylBellBasis(dimension: dimension, config: config)
        let probability = Scalar(real: Rational(1, dimension * dimension))

        let outcomes = try unitaryBasis.elements.map { element -> TeleportationOutcome in
            let measurementElement = try measurementBasis.element(for: element.index)
            let correction = element.operatorValue
            let preCorrection = try applyOperator(dagger(correction), inputState)
            let postCorrection = try applyOperator(correction, preCorrection)
            let fullPreCorrection = try tensor(
                measurementElement.state,
                preCorrection,
                config: config
            )
            let fullPostCorrection = try tensor(
                measurementElement.state,
                postCorrection,
                config: config
            )
            return TeleportationOutcome(
                index: element.index,
                measurementState: measurementElement.state,
                measurementProjector: measurementElement.projector,
                probability: probability,
                bobStateBeforeCorrection: preCorrection,
                correctionOperation: correction,
                bobStateAfterCorrection: postCorrection,
                fullStateAfterMeasurement: fullPreCorrection,
                fullStateAfterCorrection: fullPostCorrection
            )
        }

        return TeleportationScheme(
            dimension: dimension,
            inputState: inputState,
            resourceState: resource,
            initialState: initialState,
            measurementBasis: measurementBasis,
            outcomes: outcomes
        )
    }

    static func teleportationTranscript(
        inputState: Ket,
        resource: MaximallyEntangledResource,
        outcome: BellOutcome,
        config: QuantumMathConfig = .ketStepsDefault
    ) throws -> TeleportationTranscript {
        guard inputState.space.factors.count == 1,
              inputState.space.dimension == resource.dimension else {
            throw QuantumMathError.operationNotDefined(
                "Teleportation input dimension must match the resource dimension."
            )
        }
        guard inputState.basis.isComputationalCoordinateBasis else {
            throw QuantumMathError.operationNotDefined(
                "Teleportation input must use computational-coordinate basis."
            )
        }
        guard outcome.dimension == resource.dimension else {
            throw QuantumMathError.operationNotDefined(
                "Teleportation outcome dimension must match the resource dimension."
            )
        }
        try requireNormalized(inputState, config: config, purpose: "Teleportation input")

        let initialState = try tensor(inputState, resource.state, config: config)
        let measurementBasis = try resourceRelativeWeylBellBasis(resource: resource, config: config)
        let measurementElement = try measurementBasis.element(for: outcome.index)
        let transform = try bobProjectionTransform(
            measurementState: measurementElement.state,
            resourceState: resource.state,
            dimension: resource.dimension
        )
        let preCorrectionCoefficients = try transform.multiplied(by: inputState.coefficients)
        let probabilityValue = preCorrectionCoefficients.reduce(0.0) { partial, coefficient in
            partial + coefficient.magnitudeSquaredApproximate
        }
        guard probabilityValue > config.normThreshold else {
            throw QuantumMathError.operationNotDefined(
                "Selected teleportation branch has zero probability."
            )
        }

        let branchNorm = Scalar.approx(ComplexNumber(re: Foundation.sqrt(probabilityValue), im: 0))
        let normalizedPreCorrection = try preCorrectionCoefficients.map { coefficient -> Scalar in
            guard let normalized = coefficient.divided(by: branchNorm) else {
                throw QuantumMathError.operationNotDefined("Teleportation branch could not be normalized.")
            }
            return normalized
        }
        let bobStateBeforeCorrection = try Ket(
            space: inputState.space,
            basis: inputState.basis,
            coefficients: normalizedPreCorrection
        )
        let correctionScale = Scalar.approx(ComplexNumber(re: Double(resource.dimension), im: 0))
        let correctionOperation = try Operator(
            domain: inputState.space,
            codomain: inputState.space,
            columnBasis: inputState.basis,
            rowBasis: inputState.basis,
            entries: transform.conjugateTransposed().scaled(by: correctionScale)
        )
        let bobStateAfterCorrection = try normalize(
            applyOperator(correctionOperation, bobStateBeforeCorrection),
            config: config
        )
        let fullStateAfterMeasurement = try tensor(
            measurementElement.state,
            bobStateBeforeCorrection,
            config: config
        )
        let fullStateAfterCorrection = try tensor(
            measurementElement.state,
            bobStateAfterCorrection,
            config: config
        )
        let overlap = try innerProduct(inputState, bobStateAfterCorrection)
        let overlapMagnitude = Scalar.approx(ComplexNumber(re: overlap.magnitudeSquaredApproximate, im: 0))
        let succeeded = sameProjectiveRay(
            bobStateAfterCorrection,
            inputState,
            epsilon: config.scalarComparisonEpsilon
        )

        return TeleportationTranscript(
            inputState: inputState,
            resource: resource,
            initialState: initialState,
            outcome: outcome,
            measurementBasis: measurementBasis,
            measurementState: measurementElement.state,
            measurementProjector: measurementElement.projector,
            probability: Scalar.approx(ComplexNumber(re: probabilityValue, im: 0)),
            bobStateBeforeCorrection: bobStateBeforeCorrection,
            correctionOperation: correctionOperation,
            bobStateAfterCorrection: bobStateAfterCorrection,
            fullStateAfterMeasurement: fullStateAfterMeasurement,
            fullStateAfterCorrection: fullStateAfterCorrection,
            verification: ProtocolVerification(
                succeeded: succeeded,
                metric: overlapMagnitude,
                message: succeeded
                    ? "Bob's corrected state matches the input up to global phase."
                    : "Bob's corrected state does not match the input."
            )
        )
    }
}

private extension QuantumDomain {
    static func computationalBasis(
        dimension: Int,
        config: QuantumMathConfig
    ) throws -> Basis {
        try requireDimension(dimension, within: config, purpose: "Computational basis")
        let atomicSpace = try AtomicSpace(dimension: dimension)
        let space = try Space(validating: [atomicSpace], maxComputableDimension: config.maxComputableDimension)
        return .computational(for: space)
    }

    static func repeatedQuditSpace(
        dimension: Int,
        factorCount: Int,
        config: QuantumMathConfig
    ) throws -> Space {
        try requireJointDimension(
            dimension,
            factorCount: factorCount,
            within: config,
            purpose: "\(factorCount)-qudit space"
        )
        let factor = try AtomicSpace(dimension: dimension)
        return try Space(
            validating: Array(repeating: factor, count: factorCount),
            maxComputableDimension: config.maxComputableDimension
        )
    }

    static func tensorProduct(
        operators: [Operator],
        config: QuantumMathConfig
    ) throws -> Operator {
        guard let first = operators.first else {
            throw QuantumMathError.operationNotDefined("Tensor product requires at least one operator.")
        }
        return try operators.dropFirst().reduce(first) { partial, next in
            try tensor(partial, next, config: config)
        }
    }

    static func resourceRelativeWeylBellBasis(
        resource: MaximallyEntangledResource,
        config: QuantumMathConfig
    ) throws -> WeylBellBasis {
        let unitaryBasis = try weylUnitaryErrorBasis(dimension: resource.dimension, config: config)
        let elements = try unitaryBasis.elements.map { element -> WeylBellBasisElement in
            let lifted = try liftOneFactorOperator(
                element.operatorValue,
                toRawFactorOffset: 0,
                in: resource.state.space,
                config: config
            )
            let state = try applyOperator(lifted, resource.state)
            let projector = try outerProduct(state, dagger(state))
            return WeylBellBasisElement(
                index: element.index,
                state: state,
                projector: projector
            )
        }
        return try WeylBellBasis(
            dimension: resource.dimension,
            basis: resource.state.basis,
            elements: elements
        )
    }

    static func matchingBellIndex(
        for state: Ket,
        in bellBasis: WeylBellBasis,
        config: QuantumMathConfig
    ) -> WeylIndex? {
        bellBasis.elements.first { element in
            sameProjectiveRay(
                state,
                element.state,
                epsilon: config.scalarComparisonEpsilon
            )
        }?.index
    }

    static func bobProjectionTransform(
        measurementState: Ket,
        resourceState: Ket,
        dimension: Int
    ) throws -> Matrix<Scalar> {
        let values = try (0..<dimension).flatMap { bobIndex in
            try (0..<dimension).map { inputIndex in
                try (0..<dimension).reduce(Scalar.zero) { partial, aliceIndex in
                    let measurementFlat = try TensorIndexing.flatten(
                        indices: [inputIndex, aliceIndex],
                        factorDimensions: [dimension, dimension]
                    )
                    let resourceFlat = try TensorIndexing.flatten(
                        indices: [aliceIndex, bobIndex],
                        factorDimensions: [dimension, dimension]
                    )
                    return partial
                        + measurementState.coefficients[measurementFlat].conjugated
                        * resourceState.coefficients[resourceFlat]
                }
            }
        }
        return try Matrix(rows: dimension, cols: dimension, values: values)
    }

    static func sameProjectiveRay(_ lhs: Ket, _ rhs: Ket, epsilon: Double) -> Bool {
        guard lhs.space.isCoordinateCompatible(with: rhs.space),
              lhs.basis == rhs.basis,
              lhs.coefficients.count == rhs.coefficients.count else {
            return false
        }

        var factor: Scalar?
        for pair in zip(lhs.coefficients, rhs.coefficients) {
            switch (pair.0.isZero(epsilon: epsilon), pair.1.isZero(epsilon: epsilon)) {
            case (true, true):
                continue
            case (true, false), (false, true):
                return false
            case (false, false):
                guard let candidate = pair.0.divided(by: pair.1) else {
                    return false
                }
                factor = candidate
                break
            }
        }

        guard let factor else {
            return false
        }
        return zip(lhs.coefficients, rhs.coefficients).allSatisfy { pair in
            pair.0.approximatelyEquals(factor * pair.1, epsilon: epsilon)
        }
    }

    static func requireNormalized(
        _ ket: Ket,
        config: QuantumMathConfig,
        purpose: String
    ) throws {
        let normSquared = normSquared(ket)
        guard abs(normSquared - 1) <= config.scalarComparisonEpsilon else {
            throw QuantumMathError.operationNotDefined("\(purpose) must be normalized.")
        }
    }

    static func requireDimension(
        _ dimension: Int,
        within config: QuantumMathConfig,
        purpose: String
    ) throws {
        guard dimension >= 2 else {
            throw QuantumMathError.operationNotDefined("\(purpose) requires dimension >= 2.")
        }
        guard dimension <= config.maxComputableDimension else {
            throw QuantumMathError.operationNotDefined(
                "\(purpose) dimension \(dimension) exceeds configured limit \(config.maxComputableDimension)."
            )
        }
    }

    static func requireJointDimension(
        _ dimension: Int,
        factorCount: Int,
        within config: QuantumMathConfig,
        purpose: String
    ) throws {
        try requireDimension(dimension, within: config, purpose: purpose)

        var total = 1
        for _ in 0..<factorCount {
            guard dimension <= config.maxComputableDimension / total else {
                throw QuantumMathError.operationNotDefined(
                    "\(purpose) requires total dimension <= \(config.maxComputableDimension)."
                )
            }
            total *= dimension
        }
    }
}

private func positiveModulo(_ value: Int, _ modulus: Int) -> Int {
    let remainder = value % modulus
    return remainder >= 0 ? remainder : remainder + modulus
}
