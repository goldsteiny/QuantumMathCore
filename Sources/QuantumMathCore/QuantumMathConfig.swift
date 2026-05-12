import Foundation
import ExactValueRecoveryCore

public struct QuantumMathConfig: Hashable, Sendable, Codable {
    public let maxComputableDimension: Int
    public let matrixTraitEpsilon: Double
    public let scalarComparisonEpsilon: Double
    public let spectralResidualThreshold: Double
    public let svdResidualThreshold: Double
    public let projectorDistanceThreshold: Double
    public let normThreshold: Double
    public let orthogonalityThreshold: Double
    public let exactificationPolicy: ExactificationPolicy

    public init(
        maxComputableDimension: Int,
        matrixTraitEpsilon: Double,
        scalarComparisonEpsilon: Double,
        spectralResidualThreshold: Double,
        svdResidualThreshold: Double,
        projectorDistanceThreshold: Double,
        normThreshold: Double,
        orthogonalityThreshold: Double,
        exactificationPolicy: ExactificationPolicy
    ) {
        self.maxComputableDimension = maxComputableDimension
        self.matrixTraitEpsilon = matrixTraitEpsilon
        self.scalarComparisonEpsilon = scalarComparisonEpsilon
        self.spectralResidualThreshold = spectralResidualThreshold
        self.svdResidualThreshold = svdResidualThreshold
        self.projectorDistanceThreshold = projectorDistanceThreshold
        self.normThreshold = normThreshold
        self.orthogonalityThreshold = orthogonalityThreshold
        self.exactificationPolicy = exactificationPolicy
    }

    public static let ketStepsExactificationPolicy = ExactificationPolicy(
        enabledFamilies: ["rational", "radical", "complex", "rootOfUnity"],
        maxCandidatesPerVariable: 32,
        maxSearchNodes: 10_000,
        maxAcceptedSolutionsTracked: 4,
        candidateDistanceThreshold: 2.5e-7,
        mixedOutcomePolicy: .allowDomainDefined,
        ambiguityPolicy: .unresolved,
        wallClockAbortPolicy: .disabled
    )

    public static let ketStepsDefault = QuantumMathConfig(
        maxComputableDimension: 16,
        matrixTraitEpsilon: 1e-9,
        scalarComparisonEpsilon: 1e-10,
        spectralResidualThreshold: 1e-6,
        svdResidualThreshold: 1e-6,
        projectorDistanceThreshold: 1e-6,
        normThreshold: 1e-10,
        orthogonalityThreshold: 1e-10,
        exactificationPolicy: ketStepsExactificationPolicy
    )

    public static let ketStepsQutrit27Preview = QuantumMathConfig(
        maxComputableDimension: 27,
        matrixTraitEpsilon: 1e-9,
        scalarComparisonEpsilon: 1e-10,
        spectralResidualThreshold: 1e-6,
        svdResidualThreshold: 1e-6,
        projectorDistanceThreshold: 1e-6,
        normThreshold: 1e-10,
        orthogonalityThreshold: 1e-10,
        exactificationPolicy: ketStepsExactificationPolicy
    )
}
