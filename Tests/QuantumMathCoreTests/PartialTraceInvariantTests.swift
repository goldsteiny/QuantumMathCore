import Foundation
import ExactValueRecoveryCore
import Testing
@testable import QuantumMathCore

struct PartialTraceInvariantTests {
    private let config = QuantumMathConfig.ketStepsDefault
    private var matrixEpsilon: Double { config.matrixTraitEpsilon }

    @Test
    func productDensityReductionRetainsSubsystemSpaceAndFactorOrder() throws {
        let qubit = try basisState(in: .atomic(.qubit), index: 1)
        let qutrit = try basisState(in: .atomic(.qutrit), index: 2)
        let ancilla = try basisState(in: .atomic(.qubit), index: 0)

        let leftPair = try QuantumDomain.tensor(qubit, qutrit, config: config)
        let product = try QuantumDomain.tensor(leftPair, ancilla, config: config)
        let sourceDensity = try pureDensity(from: product)
        let reduced = try QuantumDomain.partialTrace(
            sourceDensity,
            tracing: try PartialTraceSelection(tracedRawOffsets: [1])
        )

        let expectedState = try QuantumDomain.tensor(qubit, ancilla, config: config)
        let expectedDensity = try pureDensity(from: expectedState)

        expectOperatorsApproximatelyEqual(reduced, expectedDensity)
        #expect(reduced.entries.isHermitian(tolerance: matrixEpsilon))
        expectDensitySemantics(reduced, purity: .pure)
    }

    @Test
    func classicallyCorrelatedMixtureReducesToMaximallyMixedMarginals() throws {
        let twoQubitSpace = try tensorSpace([.qubit, .qubit])
        let twoQubitBasis = Basis.computational(for: twoQubitSpace)

        let ket00 = try basisState(
            in: twoQubitSpace,
            basis: twoQubitBasis,
            tensorIndices: [0, 0]
        )
        let ket11 = try basisState(
            in: twoQubitSpace,
            basis: twoQubitBasis,
            tensorIndices: [1, 1]
        )
        let rho00 = try pureDensity(from: ket00)
        let rho11 = try pureDensity(from: ket11)

        let mixedValue = try QuantumDomain.weightedSum(
            terms: [(.one, .oper(rho00)), (.one, .oper(rho11))],
            normalization: .operatorUnitTrace,
            config: config
        )
        guard case let .oper(mixedDensity) = mixedValue else {
            Issue.record("Expected weighted operator sum to remain an operator.")
            return
        }

        let reducedLeft = try QuantumDomain.partialTrace(
            mixedDensity,
            tracing: try PartialTraceSelection(tracedRawOffsets: [1])
        )
        let reducedRight = try QuantumDomain.partialTrace(
            mixedDensity,
            tracing: try PartialTraceSelection(tracedRawOffsets: [0])
        )
        let expected = try maximallyMixedQubit()

        expectOperatorsApproximatelyEqual(reducedLeft, expected)
        expectOperatorsApproximatelyEqual(reducedRight, expected)
        #expect(reducedLeft.entries.isHermitian(tolerance: matrixEpsilon))
        #expect(reducedRight.entries.isHermitian(tolerance: matrixEpsilon))
        expectDensitySemantics(reducedLeft, purity: .mixed)
        expectDensitySemantics(reducedRight, purity: .mixed)
    }

    @Test
    func partialTraceIsLinearOverCompatibleTensorEndomorphisms() throws {
        let twoQubitSpace = try tensorSpace([.qubit, .qubit])
        let twoQubitBasis = Basis.computational(for: twoQubitSpace)
        let ket00 = try basisState(
            in: twoQubitSpace,
            basis: twoQubitBasis,
            tensorIndices: [0, 0]
        )
        let ket11 = try basisState(
            in: twoQubitSpace,
            basis: twoQubitBasis,
            tensorIndices: [1, 1]
        )
        let rho00 = try pureDensity(from: ket00)
        let rho11 = try pureDensity(from: ket11)
        let alpha = Scalar(real: Rational(2))
        let beta = Scalar(real: Rational(3))

        guard case let .oper(lhsInput) = try QuantumDomain.add(
            try QuantumDomain.scale(alpha, value: .oper(rho00)),
            try QuantumDomain.scale(beta, value: .oper(rho11))
        ) else {
            Issue.record("Expected scaled operator addition to remain an operator.")
            return
        }

        let lhs = try QuantumDomain.partialTrace(
            lhsInput,
            tracing: try PartialTraceSelection(tracedRawOffsets: [1])
        )
        let tracedRho00 = try QuantumDomain.partialTrace(
            rho00,
            tracing: try PartialTraceSelection(tracedRawOffsets: [1])
        )
        let tracedRho11 = try QuantumDomain.partialTrace(
            rho11,
            tracing: try PartialTraceSelection(tracedRawOffsets: [1])
        )

        guard case let .oper(rhs) = try QuantumDomain.add(
            try QuantumDomain.scale(alpha, value: .oper(tracedRho00)),
            try QuantumDomain.scale(beta, value: .oper(tracedRho11))
        ) else {
            Issue.record("Expected linear combination of traced operators to remain an operator.")
            return
        }

        expectOperatorsApproximatelyEqual(lhs, rhs)
    }

    @Test
    func partialTraceRejectsNonTensorEndomorphismWithNamedReason() throws {
        expectOperationNotDefined(containing: "tensor-product space") {
            let atomicBasis = Basis.computational(for: .atomic(.qubit))
            let identity = try Operator(
                domain: .atomic(.qubit),
                codomain: .atomic(.qubit),
                columnBasis: atomicBasis,
                rowBasis: atomicBasis,
                entries: .identity(size: 2)
            )
            return try QuantumDomain.partialTrace(
                identity,
                tracing: try PartialTraceSelection(tracedRawOffsets: [0])
            )
        }
    }

    @Test
    func partialTraceRejectsRowColumnBasisMismatchWithNamedReason() throws {
        expectOperationNotDefined(containing: "matching row and column bases") {
            let twoQubitSpace = try tensorSpace([.qubit, .qubit])
            let columnBasis = Basis.computational(for: twoQubitSpace)
            let rowBasis = try Basis(
                space: twoQubitSpace,
                factorKinds: [.hadamard, .computational]
            )
            let mismatched = try Operator(
                domain: twoQubitSpace,
                codomain: twoQubitSpace,
                columnBasis: columnBasis,
                rowBasis: rowBasis,
                entries: .identity(size: 4)
            )
            return try QuantumDomain.partialTrace(
                mismatched,
                tracing: try PartialTraceSelection(tracedRawOffsets: [0])
            )
        }
    }

    @Test
    func partialTraceRejectsInvalidFactorSelectionsWithNamedReasons() throws {
        expectOperationNotDefined(containing: "duplicates") {
            try PartialTraceSelection(tracedRawOffsets: [1, 1])
        }

        let twoQubitSpace = try tensorSpace([.qubit, .qubit])
        let basis = Basis.computational(for: twoQubitSpace)
        let identity = try Operator(
            domain: twoQubitSpace,
            codomain: twoQubitSpace,
            columnBasis: basis,
            rowBasis: basis,
            entries: .identity(size: 4)
        )

        expectOperationNotDefined(containing: "out of range") {
            try QuantumDomain.partialTrace(
                identity,
                tracing: try PartialTraceSelection(tracedRawOffsets: [2])
            )
        }
        expectOperationNotDefined(containing: "retain at least one tensor factor") {
            try QuantumDomain.partialTrace(
                identity,
                tracing: try PartialTraceSelection(tracedRawOffsets: [0, 1])
            )
        }
    }

    @Test
    func partialTraceRejectsNonSquareOperatorWithNamedReason() throws {
        let qubitSpace: Space = .atomic(.qubit)
        let qutritSpace: Space = .atomic(.qutrit)
        let rectangular = try Operator(
            domain: qubitSpace,
            codomain: qutritSpace,
            columnBasis: Basis.computational(for: qubitSpace),
            rowBasis: Basis.computational(for: qutritSpace),
            entries: try Matrix(rows: 3, cols: 2, values: [
                .one, .zero,
                .zero, .one,
                .zero, .zero
            ])
        )

        expectOperationNotDefined(containing: "square operator") {
            try QuantumDomain.partialTrace(
                rectangular,
                tracing: try PartialTraceSelection(tracedRawOffsets: [0])
            )
        }
    }

    private func tensorSpace(_ factors: [AtomicSpace]) throws -> Space {
        try Space(
            validating: factors,
            maxComputableDimension: config.maxComputableDimension
        )
    }

    private func basisState(in space: Space, index: Int) throws -> Ket {
        try basisState(in: space, basis: .computational(for: space), tensorIndices: [index])
    }

    private func basisState(
        in space: Space,
        basis: Basis,
        tensorIndices: [Int]
    ) throws -> Ket {
        let dimensions = space.factors.map(\.dimension)
        let flat = try TensorIndexing.flatten(indices: tensorIndices, factorDimensions: dimensions)
        var coefficients = Array(repeating: Scalar.zero, count: space.dimension)
        coefficients[flat] = .one
        return try Ket(space: space, basis: basis, coefficients: coefficients)
    }

    private func pureDensity(from ket: Ket) throws -> Operator {
        try QuantumDomain.pureDensityState(from: .ket(ket), config: config)
    }

    private func maximallyMixedQubit() throws -> Operator {
        let qubitSpace: Space = .atomic(.qubit)
        let basis = Basis.computational(for: qubitSpace)
        let identity = try Operator(
            domain: qubitSpace,
            codomain: qubitSpace,
            columnBasis: basis,
            rowBasis: basis,
            entries: .identity(size: qubitSpace.dimension)
        )
        guard case let .oper(scaled) = try QuantumDomain.scale(
            Scalar(real: Rational(1, 2)),
            value: .oper(identity)
        ) else {
            Issue.record("Expected scaled identity to remain an operator.")
            return identity
        }
        return scaled
    }

    private func expectOperatorsApproximatelyEqual(
        _ lhs: Operator,
        _ rhs: Operator
    ) {
        #expect(lhs.domain == rhs.domain)
        #expect(lhs.codomain == rhs.codomain)
        #expect(lhs.columnBasis == rhs.columnBasis)
        #expect(lhs.rowBasis == rhs.rowBasis)
        #expect(lhs.entries.approximatelyEquals(rhs.entries, epsilon: matrixEpsilon))
    }

    private func expectDensitySemantics(_ operatorValue: Operator, purity: DensityPurity) {
        let semantics = QuantumDomain.operatorSemantics(operatorValue, config: config)
        guard case let .density(summary) = semantics else {
            Issue.record("Expected operator semantics to classify as density.")
            return
        }
        #expect(summary.purity == purity)
        if purity == .pure {
            #expect(summary.recoveredKet != nil)
        } else {
            #expect(summary.recoveredKet == nil)
        }
    }

    private func expectOperationNotDefined<T>(
        containing fragment: String,
        _ operation: () throws -> T
    ) {
        do {
            _ = try operation()
            Issue.record("Expected operation to fail with operationNotDefined(\(fragment)).")
        } catch let QuantumMathError.operationNotDefined(message) {
            #expect(message.contains(fragment))
        } catch {
            Issue.record("Expected QuantumMathError.operationNotDefined, got \(String(describing: error)).")
        }
    }
}
