import Foundation
import ExactValueRecoveryCore
import Testing
@testable import QuantumMathCore

private struct DiagonalSingularSolver: SingularValueSolver {
    private let epsilon = 1e-10

    func decompose(_ matrix: Matrix<Scalar>) throws -> SingularValueDecompositionRaw {
        if let rankOneRaw = rankOneDecomposition(matrix) {
            return rankOneRaw
        }

        let rank = min(matrix.rows, matrix.cols)
        let singularValues = (0..<rank).map { index in
            sqrt(matrix[index, index].approximateValue.magnitudeSquared)
        }
        let left = (0..<matrix.rows).flatMap { row in
            (0..<rank).map { col in
                ComplexNumber(re: row == col ? 1 : 0, im: 0)
            }
        }
        let right = (0..<matrix.cols).flatMap { row in
            (0..<rank).map { col in
                ComplexNumber(re: row == col ? 1 : 0, im: 0)
            }
        }
        return SingularValueDecompositionRaw(
            rows: matrix.rows,
            cols: matrix.cols,
            rank: rank,
            singularValues: singularValues,
            leftVectors: left,
            rightVectors: right
        )
    }

    private func rankOneDecomposition(_ matrix: Matrix<Scalar>) -> SingularValueDecompositionRaw? {
        let rank = min(matrix.rows, matrix.cols)
        guard rank > 0 else {
            return nil
        }

        let columns = (0..<matrix.cols).map { col in
            (0..<matrix.rows).map { row in
                matrix[row, col].approximateValue
            }
        }
        guard let seedColumn = columns.first(where: { column in
            column.reduce(0.0) { partial, value in
                partial + value.magnitudeSquared
            } > epsilon
        }) else {
            return nil
        }

        let seedNorm = sqrt(seedColumn.reduce(0.0) { partial, value in
            partial + value.magnitudeSquared
        })
        guard seedNorm > epsilon else {
            return nil
        }
        let u = seedColumn.map { $0.scaled(by: 1 / seedNorm) }

        let alphas = columns.map { column in
            zip(u, column).reduce(ComplexNumber(re: 0, im: 0)) { partial, pair in
                partial + (pair.0.conjugated * pair.1)
            }
        }

        for (column, alpha) in zip(columns, alphas) {
            for (actual, basis) in zip(column, u) {
                let expected = basis * alpha
                let delta = actual - expected
                if delta.magnitudeSquared > epsilon {
                    return nil
                }
            }
        }

        let sigma = sqrt(alphas.reduce(0.0) { partial, value in
            partial + value.magnitudeSquared
        })
        guard sigma > epsilon else {
            return nil
        }
        let sigmaInverse = 1 / sigma
        let v = alphas.map { alpha in
            alpha.scaled(by: sigmaInverse).conjugated
        }

        let singularValues = [sigma] + Array(repeating: 0, count: max(0, rank - 1))
        let leftVectors = (0..<matrix.rows).flatMap { row in
            (0..<rank).map { col in
                if col == 0 {
                    return u[row]
                }
                return row == col ? ComplexNumber(re: 1, im: 0) : ComplexNumber(re: 0, im: 0)
            }
        }
        let rightVectors = (0..<matrix.cols).flatMap { row in
            (0..<rank).map { col in
                if col == 0 {
                    return v[row]
                }
                return row == col ? ComplexNumber(re: 1, im: 0) : ComplexNumber(re: 0, im: 0)
            }
        }

        return SingularValueDecompositionRaw(
            rows: matrix.rows,
            cols: matrix.cols,
            rank: rank,
            singularValues: singularValues,
            leftVectors: leftVectors,
            rightVectors: rightVectors
        )
    }
}

struct EntanglementProtocolTests {
    private let analyzer = EntanglementAnalyzer(
        config: .ketStepsQutrit27Preview,
        backend: DiagonalSingularSolver()
    )

    @Test
    func bipartitePureStateAnalysisClassifiesProductBellAndNonMaximalStates() throws {
        let product = try bipartiteKet(dimension: 2, diagonal: [.one, .zero])
        let bell = try QuantumDomain.maximallyEntangledState(
            dimension: 2,
            config: .ketStepsQutrit27Preview
        )
        let rootThreeFifths = Scalar.approx(ComplexNumber(re: sqrt(3.0 / 5.0), im: 0))
        let rootTwoFifths = Scalar.approx(ComplexNumber(re: sqrt(2.0 / 5.0), im: 0))
        let nonMaximal = try bipartiteKet(
            dimension: 2,
            diagonal: [rootThreeFifths, rootTwoFifths]
        )

        #expect(try analyzer.analyzeBipartitePureState(product).classification == .product)
        #expect(try analyzer.analyzeBipartitePureState(bell).classification == .maximallyEntangled)
        #expect(try analyzer.analyzeBipartitePureState(nonMaximal).classification == .entangled)
    }

    @Test
    func schmidtSpectrumOwnsEntanglementEntropyInBits() throws {
        let product = try analyzer.analyzeBipartitePureState(
            bipartiteKet(dimension: 2, diagonal: [.one, .zero])
        )
        let bell = try analyzer.analyzeBipartitePureState(
            QuantumDomain.maximallyEntangledState(
                dimension: 2,
                config: .ketStepsQutrit27Preview
            )
        )
        let qutritBell = try analyzer.analyzeBipartitePureState(
            QuantumDomain.maximallyEntangledState(
                dimension: 3,
                config: .ketStepsQutrit27Preview
            )
        )
        let rootThreeFifths = Scalar.approx(ComplexNumber(re: sqrt(3.0 / 5.0), im: 0))
        let rootTwoFifths = Scalar.approx(ComplexNumber(re: sqrt(2.0 / 5.0), im: 0))
        let nonMaximal = try analyzer.analyzeBipartitePureState(
            bipartiteKet(
                dimension: 2,
                diagonal: [rootThreeFifths, rootTwoFifths]
            )
        )

        let nonMaximalExpected = -((3.0 / 5.0) * log2(3.0 / 5.0))
            - ((2.0 / 5.0) * log2(2.0 / 5.0))

        #expect(product.entanglementEntropy.bits == 0)
        #expect(abs(bell.entanglementEntropy.bits - 1) <= 1e-10)
        #expect(abs(qutritBell.entanglementEntropy.bits - log2(3)) <= 1e-10)
        #expect(abs(nonMaximal.entanglementEntropy.bits - nonMaximalExpected) <= 1e-10)
    }

    @Test
    func schmidtDecompositionSupportsFactorAwarePartitionsWithResidualAndSpectrumChecks() throws {
        let state = try threeQubitStateForABSplit()
        let partition = try SchmidtFactorPartition(leftRawOffsets: [0, 1])
        let analysis = try analyzer.analyzeSchmidtDecomposition(state, partition: partition)

        #expect(analysis.partition.leftOffsets.map(\.rawValue) == [0, 1])
        #expect(analysis.partition.rightOffsets.map(\.rawValue) == [2])
        #expect(analysis.partition.leftDimension == 4)
        #expect(analysis.partition.rightDimension == 2)
        #expect(analysis.schmidtRank == 2)
        #expect(analysis.classification == .entangled)
        #expect(analysis.reconstructionResidual <= 1e-9)
        #expect(analysis.reducedSpectrumRelation.matchesWithinTolerance)
    }

    @Test
    func schmidtDecompositionDefaultPartitionUsesFirstFactorVersusRest() throws {
        let state = try threeQubitStateForABSplit()
        let analysis = try analyzer.analyzeSchmidtDecomposition(state)

        #expect(analysis.partition.leftOffsets.map(\.rawValue) == [0])
        #expect(analysis.partition.rightOffsets.map(\.rawValue) == [1, 2])
        #expect(analysis.partition.leftDimension == 2)
        #expect(analysis.partition.rightDimension == 4)
    }

    @Test
    func schmidtDecompositionUsesSchmidtKetOnRightSubsystemForComplexPhase() throws {
        let state = try productWithImaginaryRight()
        let analysis = try analyzer.analyzeSchmidtDecomposition(
            state,
            partition: SchmidtFactorPartition(leftFactors: try TensorFactorSet(rawOffsets: [0]))
        )

        #expect(analysis.schmidtRank == 1)
        #expect(analysis.components.count == 1)
        let right = analysis.components[0].rightVector.coefficients
        guard let ratio = right[1].divided(by: right[0]) else {
            Issue.record("Expected non-zero pivot coefficient in right Schmidt vector.")
            return
        }
        #expect(abs(ratio.approximateValue.re) <= 1e-8)
        #expect(abs(ratio.approximateValue.im - 1) <= 1e-8)
    }

    @Test
    func productStateExposesOnlyOneSchmidtComponent() throws {
        let product = try bipartiteKet(dimension: 2, diagonal: [.one, .zero])
        let analysis = try analyzer.analyzeSchmidtDecomposition(
            product,
            partition: SchmidtFactorPartition(leftFactors: try TensorFactorSet(rawOffsets: [0]))
        )

        #expect(analysis.schmidtRank == 1)
        #expect(analysis.schmidtCoefficients.count == 1)
        #expect(analysis.components.count == 1)
        #expect(analysis.classification == .product)
    }

    @Test
    func nonDiagonalSeparableSuperpositionRemainsRankOne() throws {
        let state = try nonDiagonalSeparableTwoQubitState()
        let analysis = try analyzer.analyzeSchmidtDecomposition(
            state,
            partition: SchmidtFactorPartition(leftFactors: try TensorFactorSet(rawOffsets: [0]))
        )

        #expect(analysis.classification == .product)
        #expect(analysis.schmidtRank == 1)
        #expect(analysis.components.count == 1)
        #expect(analysis.reconstructionResidual <= 1e-9)
    }

    @Test
    func rectangularQubitQutritStateUsesTwoSchmidtTerms() throws {
        let state = try qubitQutritEntangledState()
        let analysis = try analyzer.analyzeSchmidtDecomposition(
            state,
            partition: SchmidtFactorPartition(leftFactors: try TensorFactorSet(rawOffsets: [0]))
        )

        #expect(analysis.partition.leftDimension == 2)
        #expect(analysis.partition.rightDimension == 3)
        #expect(analysis.schmidtRank == 2)
        #expect(analysis.schmidtCoefficients.count == 2)
        #expect(analysis.components.count == 2)
        #expect(analysis.classification == .entangled)
    }

    @Test
    func schmidtDecompositionRejectsInvalidPartitions() throws {
        let state = try threeQubitStateForABSplit()
        let outOfRange = try SchmidtFactorPartition(leftRawOffsets: [3])
        let allFactors = try SchmidtFactorPartition(leftRawOffsets: [0, 1, 2])

        do {
            _ = try analyzer.analyzeSchmidtDecomposition(state, partition: outOfRange)
            Issue.record("Expected partition with out-of-range factor to fail.")
        } catch QuantumMathError.operationNotDefined(let message) {
            #expect(message.contains("out of range"))
        } catch {
            Issue.record("Expected QuantumMathError.operationNotDefined, got \(error).")
        }

        do {
            _ = try analyzer.analyzeSchmidtDecomposition(state, partition: allFactors)
            Issue.record("Expected partition that selects all factors to fail.")
        } catch QuantumMathError.operationNotDefined(let message) {
            #expect(message.contains("at least one factor"))
        } catch {
            Issue.record("Expected QuantumMathError.operationNotDefined, got \(error).")
        }
    }

    @Test
    func denseCommunicationRoundTripsEveryMessageForEveryQubitBellResource() throws {
        let bellBasis = try QuantumDomain.weylBellBasis(
            dimension: 2,
            config: .ketStepsQutrit27Preview
        )

        for element in bellBasis.elements {
            let resource = try maximallyEntangledResource(state: element.state, dimension: 2)

            for flatIndex in 0..<4 {
                let transcript = try QuantumDomain.denseCommunicationTranscript(
                    resource: resource,
                    message: try WeylMessage(dimension: 2, flatIndex: flatIndex),
                    config: .ketStepsQutrit27Preview
                )

                #expect(transcript.decodedMessage.flatIndex == flatIndex)
                #expect(transcript.verification.succeeded)
            }
        }
    }

    @Test
    func teleportationTranscriptRecoversQutritEqualSuperpositionForEveryOutcome() throws {
        let dimension = 3
        let resource = try maximallyEntangledResource(
            state: QuantumDomain.maximallyEntangledState(
                dimension: dimension,
                config: .ketStepsQutrit27Preview
            ),
            dimension: dimension
        )
        let basis = try oneQuditBasis(dimension: dimension)
        let amplitude = try QuantumDomain.inverseSquareRoot(dimension)
        let input = try Ket(
            space: basis.space,
            basis: basis,
            coefficients: Array(repeating: amplitude, count: dimension)
        )

        for flatIndex in 0..<(dimension * dimension) {
            let transcript = try QuantumDomain.teleportationTranscript(
                inputState: input,
                resource: resource,
                outcome: try BellOutcome(dimension: dimension, flatIndex: flatIndex),
                config: .ketStepsQutrit27Preview
            )

            #expect(transcript.verification.succeeded)
            #expect(sameProjectiveRay(transcript.bobStateAfterCorrection, input))
        }
    }

    private func bipartiteKet(dimension: Int, diagonal: [Scalar]) throws -> Ket {
        let space = try Space(
            validating: Array(repeating: try AtomicSpace(dimension: dimension), count: 2),
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        let basis = Basis.computational(for: space)
        var coefficients = Array(repeating: Scalar.zero, count: dimension * dimension)
        for index in 0..<dimension {
            let flat = try TensorIndexing.flatten(
                indices: [index, index],
                factorDimensions: [dimension, dimension]
            )
            coefficients[flat] = diagonal[index]
        }
        return try Ket(space: space, basis: basis, coefficients: coefficients)
    }

    private func threeQubitStateForABSplit() throws -> Ket {
        let factors = Array(repeating: AtomicSpace.qubit, count: 3)
        let space = try Space(
            validating: factors,
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        let basis = Basis.computational(for: space)
        let amplitude = Scalar.approx(ComplexNumber(re: sqrt(0.5), im: 0))
        var coefficients = Array(repeating: Scalar.zero, count: space.dimension)
        coefficients[0] = amplitude
        coefficients[3] = amplitude
        return try Ket(space: space, basis: basis, coefficients: coefficients)
    }

    private func productWithImaginaryRight() throws -> Ket {
        let space = try Space(
            validating: [.qubit, .qubit],
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        let basis = Basis.computational(for: space)
        let amplitude = Scalar.approx(ComplexNumber(re: sqrt(0.5), im: 0))
        let imaginaryAmplitude = Scalar.approx(ComplexNumber(re: 0, im: sqrt(0.5)))
        return try Ket(
            space: space,
            basis: basis,
            coefficients: [amplitude, imaginaryAmplitude, .zero, .zero]
        )
    }

    private func nonDiagonalSeparableTwoQubitState() throws -> Ket {
        let space = try Space(
            validating: [.qubit, .qubit],
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        let basis = Basis.computational(for: space)
        let quarter = Scalar.approx(ComplexNumber(re: 0.5, im: 0))
        return try Ket(
            space: space,
            basis: basis,
            coefficients: [quarter, quarter, quarter, quarter]
        )
    }

    private func qubitQutritEntangledState() throws -> Ket {
        let space = try Space(
            validating: [.qubit, try AtomicSpace(dimension: 3)],
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        let basis = Basis.computational(for: space)
        let amplitude = Scalar.approx(ComplexNumber(re: sqrt(0.5), im: 0))
        var coefficients = Array(repeating: Scalar.zero, count: space.dimension)
        coefficients[0] = amplitude
        coefficients[4] = amplitude
        return try Ket(space: space, basis: basis, coefficients: coefficients)
    }

    private func oneQuditBasis(dimension: Int) throws -> Basis {
        let space = try Space(
            validating: [try AtomicSpace(dimension: dimension)],
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        return .computational(for: space)
    }

    private func maximallyEntangledResource(
        state: Ket,
        dimension: Int
    ) throws -> MaximallyEntangledResource {
        try MaximallyEntangledResource(
            state: state,
            analysis: BipartitePureStateAnalysis(
                state: state,
                leftDimension: dimension,
                rightDimension: dimension,
                schmidtRank: dimension,
                schmidtCoefficients: Array(
                    repeating: try QuantumDomain.inverseSquareRoot(dimension),
                    count: dimension
                ),
                classification: .maximallyEntangled
            )
        )
    }

    private func sameProjectiveRay(_ lhs: Ket, _ rhs: Ket) -> Bool {
        let epsilon = 1e-8
        guard lhs.space == rhs.space,
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
                factor = pair.0.divided(by: pair.1)
            }
            if factor != nil {
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
}
