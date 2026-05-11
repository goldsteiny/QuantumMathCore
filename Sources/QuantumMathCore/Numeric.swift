import Foundation
import ExactValueRecoveryCore

public typealias ComplexNumber = ComplexApproximation

public extension ComplexApproximation {
    var conjugated: ComplexApproximation {
        ComplexApproximation(re: re, im: -im)
    }

    func scaled(by factor: Double) -> ComplexApproximation {
        ComplexApproximation(re: re * factor, im: im * factor)
    }

    static func + (lhs: ComplexApproximation, rhs: ComplexApproximation) -> ComplexApproximation {
        ComplexApproximation(re: lhs.re + rhs.re, im: lhs.im + rhs.im)
    }

    static func - (lhs: ComplexApproximation, rhs: ComplexApproximation) -> ComplexApproximation {
        ComplexApproximation(re: lhs.re - rhs.re, im: lhs.im - rhs.im)
    }

    static prefix func - (value: ComplexApproximation) -> ComplexApproximation {
        ComplexApproximation(re: -value.re, im: -value.im)
    }

    static func * (lhs: ComplexApproximation, rhs: ComplexApproximation) -> ComplexApproximation {
        ComplexApproximation(
            re: (lhs.re * rhs.re) - (lhs.im * rhs.im),
            im: (lhs.re * rhs.im) + (lhs.im * rhs.re)
        )
    }

    static func / (lhs: ComplexApproximation, rhs: ComplexApproximation) -> ComplexApproximation {
        let divisor = rhs.magnitudeSquared
        let numerator = lhs * rhs.conjugated
        return ComplexApproximation(re: numerator.re / divisor, im: numerator.im / divisor)
    }
}

public enum Scalar: Hashable, Sendable, Codable {
    case exact(ExactScalarExpression)
    case approx(ComplexApproximation)

    public static let zero = Scalar.exact(.rational(.zero))
    public static let one = Scalar.exact(.rational(.one))
    public static let i = Scalar.approx(ComplexApproximation(re: 0, im: 1))
    public static let negativeI = Scalar.approx(ComplexApproximation(re: 0, im: -1))

    public init(real: Rational) {
        self = .exact(.rational(real))
    }

    public var exactValue: ExactScalarExpression? {
        switch self {
        case let .exact(value):
            return value.canonicalized
        case .approx:
            return nil
        }
    }

    public var approximateValue: ComplexApproximation {
        switch self {
        case let .exact(value):
            return value.approximateValue
        case let .approx(value):
            return value
        }
    }

    public var isApproximate: Bool {
        switch self {
        case .exact:
            return false
        case .approx:
            return true
        }
    }

    public func isZero(epsilon: Double) -> Bool {
        let approx = approximateValue
        return abs(approx.re) <= epsilon && abs(approx.im) <= epsilon
    }

    public var magnitudeSquaredApproximate: Double {
        approximateValue.magnitudeSquared
    }

    public var conjugated: Scalar {
        let value = approximateValue.conjugated
        return .approx(value)
    }

    public func divided(by scalar: Scalar) -> Scalar? {
        if scalar.approximateValue.magnitudeSquared == 0 {
            return nil
        }
        return .approx(approximateValue / scalar.approximateValue)
    }

    public func scaled(by rational: Rational) -> Scalar {
        .approx(approximateValue.scaled(by: rational.asDouble))
    }

    public func approximatelyEquals(_ other: Scalar, epsilon: Double) -> Bool {
        let delta = approximateValue - other.approximateValue
        return abs(delta.re) <= epsilon && abs(delta.im) <= epsilon
    }

    public static prefix func - (value: Scalar) -> Scalar {
        .approx(-value.approximateValue)
    }

    public static func + (lhs: Scalar, rhs: Scalar) -> Scalar {
        .approx(lhs.approximateValue + rhs.approximateValue)
    }

    public static func - (lhs: Scalar, rhs: Scalar) -> Scalar {
        .approx(lhs.approximateValue - rhs.approximateValue)
    }

    public static func * (lhs: Scalar, rhs: Scalar) -> Scalar {
        .approx(lhs.approximateValue * rhs.approximateValue)
    }
}
