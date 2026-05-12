import Foundation
import ExactValueRecoveryCore
import Testing
@testable import QuantumMathCore

struct ExactificationAdapterTests {
    @Test
    func scalarExactificationRecoversSimpleRational() {
        let result = ScalarExactificationAdapter.exactify(
            .approx(ComplexNumber(re: 0.5, im: 0)),
            label: "unit-test",
            config: .ketStepsDefault
        )

        #expect(!result.scalar.isApproximate)
    }

    @Test
    func stateVectorExactificationRecoversSimpleRadicalCoefficients() throws {
        let space = try Space(validating: [.qubit], maxComputableDimension: 16)
        let basis = Basis.computational(for: space)
        let inverseRootTwo = 1 / Foundation.sqrt(2)
        let approximatePlus = try Ket(
            space: space,
            basis: basis,
            coefficients: [
                .approx(ComplexNumber(re: inverseRootTwo, im: 0)),
                .approx(ComplexNumber(re: inverseRootTwo, im: 0))
            ]
        )

        let result = StateVectorExactificationAdapter.exactify(
            approximatePlus,
            label: "plus",
            config: .ketStepsDefault
        )

        #expect(result.ket.coefficients.allSatisfy { !$0.isApproximate })
    }

    @Test
    func exactificationCacheReusesCommonScalarHits() {
        let cache = QuantumExactificationCache()
        let scalar = Scalar.approx(ComplexNumber(re: 1 / Foundation.sqrt(2), im: 0))

        let first = cache.exactify(scalar, config: strictExactificationConfig())
        let afterFirst = cache.stats
        let second = cache.exactify(scalar, config: strictExactificationConfig())
        let afterSecond = cache.stats

        #expect(!first.isApproximate)
        #expect(!second.isApproximate)
        #expect(afterFirst.lookupMisses == 1)
        #expect(afterFirst.storedExactValues == 1)
        #expect(afterSecond.lookupHits == 1)
        #expect(afterSecond.storedExactValues == 1)
    }

    @Test
    func exactificationCacheStoresUnsupportedMissesWithoutSnapping() {
        let cache = QuantumExactificationCache()
        let unsupported = Scalar.approx(ComplexNumber(re: 1.000_001, im: 0))
        let config = strictExactificationConfig()

        let first = cache.exactify(unsupported, config: config)
        let afterFirst = cache.stats
        let second = cache.exactify(unsupported, config: config)
        let afterSecond = cache.stats

        #expect(first.isApproximate)
        #expect(second.isApproximate)
        #expect(afterFirst.lookupMisses == 1)
        #expect(afterFirst.storedMisses == 1)
        #expect(afterSecond.lookupHits == 1)
        #expect(afterSecond.storedMisses == 1)
    }

    @Test
    func exactificationCacheKeysIncludeToleranceProfile() {
        let cache = QuantumExactificationCache()
        let scalar = Scalar.approx(ComplexNumber(re: 1 / Foundation.sqrt(2), im: 0))

        _ = cache.exactify(scalar, config: strictExactificationConfig(threshold: 2.5e-7))
        let afterFirst = cache.stats
        _ = cache.exactify(scalar, config: strictExactificationConfig(threshold: 1.0e-8))
        let afterSecond = cache.stats

        #expect(afterFirst.lookupMisses == 1)
        #expect(afterSecond.lookupMisses == 2)
        #expect(afterSecond.storedExactValues == 2)
    }

    private func strictExactificationConfig(threshold: Double = 2.5e-7) -> QuantumMathConfig {
        QuantumMathConfig(
            maxComputableDimension: 16,
            matrixTraitEpsilon: 1e-9,
            scalarComparisonEpsilon: 1e-10,
            spectralResidualThreshold: 1e-6,
            svdResidualThreshold: 1e-6,
            projectorDistanceThreshold: 1e-6,
            normThreshold: 1e-10,
            orthogonalityThreshold: 1e-10,
            exactificationPolicy: ExactificationPolicy(
                enabledFamilies: ["rational", "radical", "complex", "rootOfUnity"],
                maxCandidatesPerVariable: 32,
                maxSearchNodes: 10_000,
                maxAcceptedSolutionsTracked: 4,
                candidateDistanceThreshold: threshold,
                mixedOutcomePolicy: .allowDomainDefined,
                ambiguityPolicy: .unresolved,
                wallClockAbortPolicy: .disabled
            )
        )
    }
}
