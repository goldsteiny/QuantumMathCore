import Foundation
import ExactValueRecoveryCore
import Testing
@testable import QuantumMathCore

private struct DiagonalSingularSolver: SingularValueSolver {
    func decompose(_ matrix: Matrix<Scalar>) throws -> SingularValueDecompositionRaw {
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
