// LandmarkSample.swift
import Foundation
import MediaPipeTasksVision

/// Tipo semplice e Codable per serializzare i singoli punti landmark
public struct LandmarkPoint: Codable {
    public let x: Float
    public let y: Float
    public let z: Float
    public let visibility: Float? // probabilmente nil per hand landmarks
    public let presence: Float?   // probabilmente nil per hand landmarks

    public init(x: Float, y: Float, z: Float, visibility: Float? = nil, presence: Float? = nil) {
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence
    }

    /// Inizializza da NormalizedLandmark di MediaPipe
    public init(from normalized: NormalizedLandmark) {
        self.x = normalized.x
        self.y = normalized.y
        self.z = normalized.z
        self.visibility = normalized.visibility?.floatValue
        self.presence = normalized.presence?.floatValue
    }
}

// MARK: - Hashable per LandmarkPoint
extension LandmarkPoint: Hashable {
    public func hash(into hasher: inout Hasher) {
        // Arrotondamento leggero per evitare problemi di float minimi
        hasher.combine(Int(x * 1000))
        hasher.combine(Int(y * 1000))
        hasher.combine(Int(z * 1000))
    }

    public static func == (lhs: LandmarkPoint, rhs: LandmarkPoint) -> Bool {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z
    }
}

/// Landmark sample con label e lista di landmarks, eventualmente anche features
public struct LandmarkSample: Codable {
    public let label: String
    public let landmarks: [LandmarkPoint]
    public let features: [Double]?

    public init(label: String, landmarks: [LandmarkPoint]) {
        self.label = label
        self.landmarks = landmarks
        self.features = nil
    }

    public init(label: String, landmarks: [LandmarkPoint], features: [Double]) {
        self.label = label
        self.landmarks = landmarks
        self.features = features
    }
}

// MARK: - Hashable per LandmarkSample
extension LandmarkSample: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(label.lowercased()) // case-insensitive
        for landmark in landmarks {
            hasher.combine(landmark)
        }
    }

    public static func == (lhs: LandmarkSample, rhs: LandmarkSample) -> Bool {
        return lhs.label.lowercased() == rhs.label.lowercased() &&
               lhs.landmarks == rhs.landmarks
    }
}
