import Foundation
import Accelerate

/// Utility functions for landmark processing and feature extraction with performance optimizations
struct LandmarkUtils {
    static var selectedFeatureIndices: [Int]?
    
    // Cache for computed features with size limit
    private static var featureCache: [String: [Double]] = [:]
    private static let cacheQueue = DispatchQueue(label: "com.landmarkutils.featurecache", attributes: .concurrent)
    private static let maxCacheSize = 100
    
    /// Prepares a sample for training with optional feature selection and caching
    /// - Parameters:
    ///   - sample: The LandmarkSample to process
    ///   - useFeatureSelection: Whether to apply feature selection
    /// - Returns: Array of feature values
    static func prepareForTraining(sample: LandmarkSample, useFeatureSelection: Bool = false) -> [Double] {
        let cacheKey = "\(sample.label)_\(sample.landmarks.hashValue)"

        if let cachedFeatures = getCachedFeatures(for: cacheKey) {
            return useFeatureSelection ? applyFeatureSelection(cachedFeatures) : cachedFeatures
        }
        
        let augmentedSample = augment(sample: sample)
        let normalized = normalizedFlatVector(from: augmentedSample)
        let geometricFeatures = extractGeometricFeatures(from: augmentedSample)
        let fullFeatures = normalized + geometricFeatures
        
        // DEBUG: Stampa il numero di feature
        print("DEBUG: Training features - Normalized: \(normalized.count), Geometric: \(geometricFeatures.count), Total: \(fullFeatures.count)")
        
        cacheFeatures(fullFeatures, for: cacheKey)
        
        return useFeatureSelection ? applyFeatureSelection(fullFeatures) : fullFeatures
    }
    
    /// Applies feature selection to a feature vector
    /// - Parameter features: Original feature vector
    /// - Returns: Filtered feature vector using selected indices
    static func applyFeatureSelection(_ features: [Double]) -> [Double] {
        guard let indices = selectedFeatureIndices else {
            return features
        }
        return indices.map { features[$0] }
    }
    
    /// Prepares landmarks for prediction with feature selection
    /// - Parameter landmarks: Array of LandmarkPoints
    /// - Returns: Feature vector ready for prediction
    static func prepareForPrediction(from landmarks: [LandmarkPoint]) -> [Double] {
        // Usa la stessa normalizzazione usata nel training
        let normalized = normalizedFlatVector(landmarks: landmarks)
        
        // Estrai features geometriche (crea un sample temporaneo)
        let tempSample = LandmarkSample(label: "temp", landmarks: landmarks)
        let geometricFeatures = extractGeometricFeatures(from: tempSample)
        
        // Combina le features
        let allFeatures = normalized + geometricFeatures
        
        // Applica feature selection se disponibile
        if let indices = selectedFeatureIndices {
            return indices.map { allFeatures[$0] }
        }
        
        return allFeatures
    }
    static func normalizeLandmarks(_ landmarks: [(x: Double, y: Double, z: Double)]) -> [(x: Double, y: Double, z: Double)] {
        return normalizedFlatVector(landmarks: landmarks.map {
            LandmarkPoint(x: Float($0.x), y: Float($0.y), z: Float($0.z))
        }).chunked(into: 3).map { chunk in
            (x: chunk[0], y: chunk[1], z: chunk[2])
        }
    }
    
    static func verifyFeatureConsistency() {
        // Crea un sample di test
        let testLandmarks = Array(repeating: LandmarkPoint(x: 0.5, y: 0.5, z: 0.5), count: 21)
        let testSample = LandmarkSample(label: "test", landmarks: testLandmarks)
        
        // Estrai le feature come nel training
        let normalized = normalizedFlatVector(from: testSample)
        let geometricFeatures = extractGeometricFeatures(from: testSample)
        let totalFeatures = normalized + geometricFeatures
        
        print("DEBUG: Actual feature counts - Normalized: \(normalized.count), Geometric: \(geometricFeatures.count), Total: \(totalFeatures.count)")
        
        // Verifica gli indici selezionati
        if let indices = selectedFeatureIndices {
            let validIndices = indices.filter { $0 >= 0 && $0 < totalFeatures.count }
            print("DEBUG: Selected indices validation - Total: \(indices.count), Valid: \(validIndices.count)")
            
            if validIndices.count != indices.count {
                print("ATTENZIONE: \(indices.count - validIndices.count) indices are out of range")
            }
        }
    }
   
    static func debugFileLocations() {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        print("Documents directory: \(documentsURL.path)")
        
        for bundle in Bundle.allBundles + Bundle.allFrameworks {
            if let path = bundle.resourcePath {
                print("Bundle: \(bundle.bundleIdentifier ?? "sconosciuto") - \(path)")
            }
        }
    }
    static func loadSelectedFeatureIndices() -> [Int]? {
        // Prima controlla UserDefaults
        if let indices = UserDefaults.standard.array(forKey: "selectedFeatureIndices") as? [Int] {
               selectedFeatureIndices = indices
               print("Caricati \(indices.count) indici da UserDefaults")
               
               // Verifica che gli indici siano validi
               let expectedFeatureCount = 63 + 42
               let validIndices = indices.filter { $0 >= 0 && $0 < expectedFeatureCount }
               
               if validIndices.count != indices.count {
                   print("ATTENZIONE: \(indices.count - validIndices.count) indici sono fuori range all'avvio")
                   selectedFeatureIndices = validIndices
                   UserDefaults.standard.set(validIndices, forKey: "selectedFeatureIndices")
               }
               
               return selectedFeatureIndices
           }
        
        // Poi controlla il file JSON nel documents directory
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent("selected_features.json")
        
        if FileManager.default.fileExists(atPath: fileURL.path),
           let data = try? Data(contentsOf: fileURL),
           let indices = try? JSONDecoder().decode([Int].self, from: data) {
            selectedFeatureIndices = indices
            print("Caricati \(indices.count) indici dal file JSON in Documents")
            return indices
        }
        
        // Infine controlla il bundle come fallback
        for bundle in Bundle.allBundles + Bundle.allFrameworks {
            if let url = bundle.url(forResource: "selected_features", withExtension: "json"),
               let data = try? Data(contentsOf: url),
               let indices = try? JSONDecoder().decode([Int].self, from: data) {
                selectedFeatureIndices = indices
                print("Caricati \(indices.count) indici dal bundle")
                return indices
            }
        }
        
        print("Nessun indice trovato")
        return nil
    }
    
    static func loadIndicesFromURL(_ url: URL) -> [Int]? {
        do {
            let data = try Data(contentsOf: url)
            let indices = try JSONDecoder().decode([Int].self, from: data)
            print("Caricati \(indices.count) indici di feature selezionate")
            print("Indici: \(indices)")
            return indices
        } catch {
            print("Errore nel caricamento degli indici da \(url): \(error)")
            return nil
        }
    }
    /// Saves selected feature indices to UserDefaults
    /// - Parameter indices: Array of indices to save
    static func saveSelectedFeatureIndices(_ indices: [Int]) {
        // Verifica che gli indici siano validi rispetto al numero atteso di feature
        let expectedFeatureCount = 63 + 42 // 63 normalized + 42 geometric
        let validIndices = indices.filter { $0 >= 0 && $0 < expectedFeatureCount }
        
        if validIndices.count != indices.count {
            print("ATTENZIONE: \(indices.count - validIndices.count) indici sono fuori range durante il salvataggio")
        }
        
        selectedFeatureIndices = validIndices
        UserDefaults.standard.set(validIndices, forKey: "selectedFeatureIndices")
        
        // Salva anche nel file JSON
        do {
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let fileURL = documentsURL.appendingPathComponent("selected_features.json")
            let data = try JSONEncoder().encode(validIndices)
            try data.write(to: fileURL)
            print("Indici salvati: \(validIndices)")
        } catch {
            print("Errore nel salvataggio degli indici: \(error)")
        }
    }
   
    
    /// Applies appropriate augmentation based on sample label
    /// - Parameter sample: Original LandmarkSample
    /// - Returns: Augmented LandmarkSample
    static func augment(sample: LandmarkSample) -> LandmarkSample {
        let label = sample.label.lowercased()
        if label.contains("thumb") || label.contains("up") || label.contains("down") {
            return conservativeAugmentation(for: sample)
        } else {
            return advancedAugment(sample: sample)
        }
    }
    
    
    /// Extracts geometric features from a sample
    /// - Parameter sample: LandmarkSample to process
    /// - Returns: Array of geometric feature values
    static func extractGeometricFeatures(from sample: LandmarkSample) -> [Double] {
        let landmarks = sample.landmarks
        var features: [Double] = []
        
        // Base features (13 features)
        features.append(contentsOf: calculateKeyDistances(landmarks: landmarks)) // 5 features
        features.append(contentsOf: calculateJointAngles(landmarks: landmarks)) // 1 feature
        features.append(contentsOf: calculateInterFingerDistances(landmarks: landmarks)) // 4 features
        features.append(calculateHandSpreadAngle(landmarks: landmarks)) // 1 feature
        features.append(calculateAverageFingerLength(landmarks: landmarks)) // 1 feature
        features.append(calculateThumbIndexLengthRatio(landmarks: landmarks)) // 1 feature
        
        // Finger features (25 features)
        features.append(contentsOf: extractFingerFeatures(landmarks: landmarks)) // 5 fingers × 5 features
        
        // Global hand features (4 features)
        features.append(contentsOf: extractGlobalHandFeatures(landmarks: landmarks)) // 4 features
        
        return features
    }
    
    // MARK: - Private Methods
    
    static func normalizedFlatVector(from sample: LandmarkSample) -> [Double] {
        return normalizedFlatVector(landmarks: sample.landmarks)
    }

    private static func normalizedFlatVector(landmarks: [LandmarkPoint]) -> [Double] {
        guard !landmarks.isEmpty else { return [] }
        
        let (cx, cy, cz) = calculateCentroid(of: landmarks)
        let maxDistance = calculateMaxDistance(from: landmarks, cx: cx, cy: cy, cz: cz)
        
        return landmarks.flatMap { landmark in
            [
                (Double(landmark.x) - cx) / maxDistance,
                (Double(landmark.y) - cy) / maxDistance,
                (Double(landmark.z) - cz) / maxDistance
            ]
        }
    }
    
    static func calculateCentroid(of landmarks: [LandmarkPoint]) -> (Double, Double, Double) {
       var xSum = 0.0, ySum = 0.0, zSum = 0.0
       let count = Double(landmarks.count)
       
       // Usa un ciclo ottimizzato
       for landmark in landmarks {
           xSum += Double(landmark.x)
           ySum += Double(landmark.y)
           zSum += Double(landmark.z)
       }
       
       return (xSum / count, ySum / count, zSum / count)
   }
    
    private static func calculateMaxDistance(from landmarks: [LandmarkPoint],
                                           cx: Double, cy: Double, cz: Double) -> Double {
        var maxDist = 0.0
        
        for landmark in landmarks {
            let dx = Double(landmark.x) - cx
            let dy = Double(landmark.y) - cy
            let dz = Double(landmark.z) - cz
            
            let distance = sqrt(dx*dx + dy*dy + dz*dz)
            maxDist = max(maxDist, distance)
        }
        
        return maxDist > 0 ? maxDist : 1.0
    }
    
    // MARK: - Distance Calculations
    
    private static func calculateKeyDistances(landmarks: [LandmarkPoint]) -> [Double] {
        guard let wrist = landmarks.first else { return [] }
        let fingertipIndices = [4, 8, 12, 16, 20]
        
        return fingertipIndices.compactMap { index in
            guard index < landmarks.count else { return nil }
            return distanceBetween(wrist, landmarks[index])
        }
    }
    
    private static func calculateInterFingerDistances(landmarks: [LandmarkPoint]) -> [Double] {
        let pairs = [(4,8), (8,12), (12,16), (16,20)]
        return pairs.compactMap { (i,j) in
            guard i < landmarks.count, j < landmarks.count else { return nil }
            return distanceBetween(landmarks[i], landmarks[j])
        }
    }
    
    private static func distanceBetween(_ p1: LandmarkPoint, _ p2: LandmarkPoint) -> Double {
        let dx = Double(p1.x - p2.x)
        let dy = Double(p1.y - p2.y)
        let dz = Double(p1.z - p2.z)
        return sqrt(dx*dx + dy*dy + dz*dz)
    }
    
    // MARK: - Angle Calculations
    
    private static func calculateJointAngles(landmarks: [LandmarkPoint]) -> [Double] {
        guard landmarks.count >= 5 else { return [] }
        
        let angle = angleBetweenVectors(
            from: landmarks[1],
            to: landmarks[0],
            from: landmarks[1],
            to: landmarks[4]
        )
        return [angle]
    }
    
    private static func calculateHandSpreadAngle(landmarks: [LandmarkPoint]) -> Double {
        guard landmarks.count > 20 else { return 0.0 }
        return angleBetweenVectors(
            from: landmarks[0], to: landmarks[5],
            from: landmarks[0], to: landmarks[17]
        )
    }
    
    private static func angleBetweenVectors(from point1: LandmarkPoint, to point2: LandmarkPoint,
                                          from point3: LandmarkPoint, to point4: LandmarkPoint) -> Double {
        let vector1 = (x: Double(point2.x - point1.x),
                      y: Double(point2.y - point1.y),
                      z: Double(point2.z - point1.z))
        let vector2 = (x: Double(point4.x - point3.x),
                      y: Double(point4.y - point3.y),
                      z: Double(point4.z - point3.z))
        
        let dotProduct = vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z
        let magnitude1 = sqrt(vector1.x * vector1.x + vector1.y * vector1.y + vector1.z * vector1.z)
        let magnitude2 = sqrt(vector2.x * vector2.x + vector2.y * vector2.y + vector2.z * vector2.z)
        
        let cosine = dotProduct / (magnitude1 * magnitude2)
        return acos(max(-1, min(1, cosine)))
    }
    
    // MARK: - Length Calculations
    
    private static func calculateAverageFingerLength(landmarks: [LandmarkPoint]) -> Double {
        let wrist = landmarks[0]
        let fingertipIndices = [8, 12, 16]
        let lengths = fingertipIndices.compactMap { i in
            i < landmarks.count ? distanceBetween(wrist, landmarks[i]) : nil
        }
        
        guard !lengths.isEmpty else { return 0.0 }
        
        var sum = 0.0
        vDSP_sveD(lengths, 1, &sum, vDSP_Length(lengths.count))
        return sum / Double(lengths.count)
    }
    
    private static func calculateThumbIndexLengthRatio(landmarks: [LandmarkPoint]) -> Double {
        guard landmarks.count > 8 else { return 1.0 }
        let wrist = landmarks[0]
        let thumbLength = distanceBetween(wrist, landmarks[4])
        let indexLength = distanceBetween(wrist, landmarks[8])
        return indexLength > 0 ? thumbLength / indexLength : 1.0
    }
    
    // MARK: - Finger Features
    
    private static func extractFingerFeatures(landmarks: [LandmarkPoint]) -> [Double] {
        var allFingerFeatures: [Double] = []
        
        let fingerIndices = [
            [1, 2, 3, 4],       // Thumb
            [5, 6, 7, 8],       // Index
            [9, 10, 11, 12],    // Middle
            [13, 14, 15, 16],   // Ring
            [17, 18, 19, 20]    // Pinky
        ]
        
        for finger in fingerIndices {
            guard finger.count == 4, landmarks.count > finger[3] else { continue }
            
            let base = landmarks[finger[0]]
            let tip = landmarks[finger[3]]
            
            let directionX = Double(tip.x - base.x)
            let directionY = Double(tip.y - base.y)
            let angle = atan2(directionY, directionX)
            let length = sqrt(directionX * directionX + directionY * directionY)
            
            let curvature = calculateFingerCurvature(landmarks: landmarks, indices: finger)
            
            allFingerFeatures.append(contentsOf: [directionX, directionY, angle, length, curvature])
        }
        
        return allFingerFeatures
    }
    
    private static func calculateFingerCurvature(landmarks: [LandmarkPoint], indices: [Int]) -> Double {
        guard indices.count == 4 else { return 0.0 }
        
        let p0 = landmarks[indices[0]]
        let p1 = landmarks[indices[1]]
        let p2 = landmarks[indices[2]]
        let p3 = landmarks[indices[3]]
        
        let directDistance = distanceBetween(p0, p3)
        let segmentDistance = distanceBetween(p0, p1) + distanceBetween(p1, p2) + distanceBetween(p2, p3)
        
        return segmentDistance / directDistance
    }
    
    // MARK: - Global Hand Features
    
    private static func extractGlobalHandFeatures(landmarks: [LandmarkPoint]) -> [Double] {
        guard landmarks.count >= 21 else { return [] }
        
        var features: [Double] = []
        
        // 1. Hand bounding box area
        let xs = landmarks.map { Double($0.x) }
        let ys = landmarks.map { Double($0.y) }
        
        var minX = 0.0, maxX = 0.0, minY = 0.0, maxY = 0.0
        vDSP_minvD(xs, 1, &minX, vDSP_Length(xs.count))
        vDSP_maxvD(xs, 1, &maxX, vDSP_Length(xs.count))
        vDSP_minvD(ys, 1, &minY, vDSP_Length(ys.count))
        vDSP_maxvD(ys, 1, &maxY, vDSP_Length(ys.count))
        
        let handArea = (maxX - minX) * (maxY - minY)
        features.append(handArea)
        
        // 2. Hand aspect ratio (width/height)
        let handAspectRatio = (maxX - minX) / (maxY - minY)
        features.append(handAspectRatio)
        
        // 3. Center of mass offset relative to wrist
        let wrist = landmarks[0]
        let (cx, cy, _) = calculateCentroid(of: landmarks)
        let centerOffsetX = cx - Double(wrist.x)
        let centerOffsetY = cy - Double(wrist.y)
        features.append(contentsOf: [centerOffsetX, centerOffsetY])
        
        return features
    }
    
    // MARK: - Cache Management
    
    private static func getCachedFeatures(for key: String) -> [Double]? {
        return cacheQueue.sync {
            return featureCache[key]
        }
    }
    
    private static func cacheFeatures(_ features: [Double], for key: String) {
        cacheQueue.async(flags: .barrier) {
            featureCache[key] = features
            
            // Limit cache size to prevent memory issues
            if featureCache.count > maxCacheSize {
                // Remove the oldest entry (assuming keys are added in order)
                if let firstKey = featureCache.keys.first {
                    featureCache.removeValue(forKey: firstKey)
                }
            }
        }
    }
    
    // MARK: - Augmentation Methods (remain the same)
    
    private static func advancedAugment(sample: LandmarkSample) -> LandmarkSample {
        // Implementation unchanged from original
        var augmentedLandmarks = sample.landmarks
        
        switch Int.random(in: 0...4) {
        case 0:
            augmentedLandmarks = addGaussianNoise(to: augmentedLandmarks, mean: 0, stdDev: 0.01)
        case 1:
            augmentedLandmarks = applyNonUniformScaling(to: augmentedLandmarks, scaleRange: (0.9, 1.1))
        case 2:
            augmentedLandmarks = apply3DRotation(to: augmentedLandmarks, maxAngle: 0.05)
        case 3:
            augmentedLandmarks = applyTranslation(to: augmentedLandmarks, maxOffset: 0.03)
        default:
            augmentedLandmarks = addGaussianNoise(to: augmentedLandmarks, mean: 0, stdDev: 0.008)
            augmentedLandmarks = applyNonUniformScaling(to: augmentedLandmarks, scaleRange: (0.95, 1.05))
        }
        
        return LandmarkSample(label: sample.label, landmarks: augmentedLandmarks)
    }
    
    private static func conservativeAugmentation(for sample: LandmarkSample) -> LandmarkSample {
        // Implementation unchanged from original
        var landmarks = sample.landmarks
        
        switch Int.random(in: 0...2) {
        case 0:
            landmarks = addGaussianNoise(to: landmarks, mean: 0, stdDev: 0.005)
        case 1:
            landmarks = applyUniformScaling(to: landmarks, scaleRange: (0.97, 1.03))
        default:
            landmarks = applyTranslation(to: landmarks, maxOffset: 0.01)
        }
        
        return LandmarkSample(label: sample.label, landmarks: landmarks)
    }
    
    // MARK: - Atomic Transformations (remain the same)
    
    private static func addGaussianNoise(to landmarks: [LandmarkPoint],
                                       mean: Double, stdDev: Double) -> [LandmarkPoint] {
        // Implementation unchanged from original
        return landmarks.map { landmark in
            let noiseX = Double.random(in: -stdDev...stdDev) + mean
            let noiseY = Double.random(in: -stdDev...stdDev) + mean
            let noiseZ = Double.random(in: -stdDev...stdDev) + mean
            
            return LandmarkPoint(
                x: Float(Double(landmark.x) + noiseX),
                y: Float(Double(landmark.y) + noiseY),
                z: Float(Double(landmark.z) + noiseZ)
            )
        }
    }
    
    private static func applyNonUniformScaling(to landmarks: [LandmarkPoint],
                                             scaleRange: (Double, Double)) -> [LandmarkPoint] {
        // Implementation unchanged from original
        let centerX = landmarks.map { Double($0.x) }.reduce(0, +) / Double(landmarks.count)
        let centerY = landmarks.map { Double($0.y) }.reduce(0, +) / Double(landmarks.count)
        
        return landmarks.map { landmark in
            let scaleX = Double.random(in: scaleRange.0...scaleRange.1)
            let scaleY = Double.random(in: scaleRange.0...scaleRange.1)
            
            let scaledX = centerX + (Double(landmark.x) - centerX) * scaleX
            let scaledY = centerY + (Double(landmark.y) - centerY) * scaleY
            
            return LandmarkPoint(x: Float(scaledX), y: Float(scaledY), z: landmark.z)
        }
    }
    
    private static func applyUniformScaling(to landmarks: [LandmarkPoint],
                                          scaleRange: (Double, Double)) -> [LandmarkPoint] {
        // Implementation unchanged from original
        let centerX = landmarks.map { Double($0.x) }.reduce(0, +) / Double(landmarks.count)
        let centerY = landmarks.map { Double($0.y) }.reduce(0, +) / Double(landmarks.count)
        let scale = Double.random(in: scaleRange.0...scaleRange.1)
        
        return landmarks.map { landmark in
            let scaledX = centerX + (Double(landmark.x) - centerX) * scale
            let scaledY = centerY + (Double(landmark.y) - centerY) * scale
            
            return LandmarkPoint(x: Float(scaledX), y: Float(scaledY), z: landmark.z)
        }
    }
    
    private static func apply3DRotation(to landmarks: [LandmarkPoint], maxAngle: Double) -> [LandmarkPoint] {
        // Implementation unchanged from original
        let (cx, cy, cz) = calculateCentroid(of: landmarks)
        
        let angleX = Double.random(in: -maxAngle...maxAngle)
        let angleY = Double.random(in: -maxAngle...maxAngle)
        let angleZ = Double.random(in: -maxAngle...maxAngle)
        
        return landmarks.map { landmark in
            let translatedX = Double(landmark.x) - cx
            let translatedY = Double(landmark.y) - cy
            let translatedZ = Double(landmark.z) - cz
            
            // X-axis rotation
            let y1 = translatedY * cos(angleX) - translatedZ * sin(angleX)
            let z1 = translatedY * sin(angleX) + translatedZ * cos(angleX)
            
            // Y-axis rotation
            let x2 = translatedX * cos(angleY) + z1 * sin(angleY)
            let z2 = -translatedX * sin(angleY) + z1 * cos(angleY)
            
            // Z-axis rotation
            let x3 = x2 * cos(angleZ) - y1 * sin(angleZ)
            let y3 = x2 * sin(angleZ) + y1 * cos(angleZ)
            
            return LandmarkPoint(
                x: Float(x3 + cx),
                y: Float(y3 + cy),
                z: Float(z2 + cz)
            )
        }
    }
    
    private static func applyTranslation(to landmarks: [LandmarkPoint],
                                       maxOffset: Double) -> [LandmarkPoint] {
        // Implementation unchanged from original
        let offsetX = Double.random(in: -maxOffset...maxOffset)
        let offsetY = Double.random(in: -maxOffset...maxOffset)
        
        return landmarks.map { landmark in
            LandmarkPoint(
                x: Float(Double(landmark.x) + offsetX),
                y: Float(Double(landmark.y) + offsetY),
                z: landmark.z
            )
        }
    }
    
    //========DEBUG==========
    static func verifyNormalizationConsistency() {
        let testLandmarks = (0..<21).map { i in
            LandmarkPoint(x: Float(0.1 * Double(i)), y: Float(0.2 * Double(i)), z: Float(0.3 * Double(i)))
        }
        
        // Usa lo stesso metodo per entrambi
        let trainingNormalized = normalizedFlatVector(landmarks: testLandmarks)
        let predictionNormalized = normalizedFlatVector(landmarks: testLandmarks)
        
        print("[DEBUGGING] Training normalized: \(trainingNormalized.prefix(5))...")
        print("[DEBUGGING] Prediction normalized: \(predictionNormalized.prefix(5))...")
        
        let isConsistent = zip(trainingNormalized, predictionNormalized).allSatisfy { abs($0 - $1) < 1e-10 }
        print("[DEBUGGING] Normalization consistent: \(isConsistent)")
    }

    static func verifyFeatureIndices() {
        guard let indices = selectedFeatureIndices else {
            print("[DEBUGGING] Nessun indice di feature selezionate")
            return
        }
        
        print("[DEBUGGING] Indici feature selezionate: \(indices)")
        print("[DEBUGGING] Range indici: \(indices.min() ?? 0) - \(indices.max() ?? 0)")
        
        // Verifica che non ci siano duplicati
        let uniqueIndices = Set(indices)
        if uniqueIndices.count != indices.count {
            print("[DEBUGGING] ATTENZIONE: \(indices.count - uniqueIndices.count) indici duplicati")
        }
        
        // Verifica che gli indici siano nell'intervallo corretto
        let outOfRange = indices.filter { $0 < 0 || $0 >= 105 }
        if !outOfRange.isEmpty {
            print("[DEBUGGING] ATTENZIONE: \(outOfRange.count) indici fuori range: \(outOfRange)")
        }
    }
    
    // Aggiungi a LandmarkUtils
    static func validateLandmarks(_ landmarks: [LandmarkPoint]) -> Bool {
        guard landmarks.count == 21 else {
            print("[DEBUGGING] ERRORE: Numero di landmark errato: \(landmarks.count) invece di 21")
            return false
        }
        
        // Controlla per valori NaN o infiniti
        for (i, landmark) in landmarks.enumerated() {
            if landmark.x.isNaN || landmark.x.isInfinite ||
               landmark.y.isNaN || landmark.y.isInfinite ||
               landmark.z.isNaN || landmark.z.isInfinite {
                print("[DEBUGGING] ERRORE: Landmark \(i) contiene valori non validi: \(landmark)")
                return false
            }
        }
        
        // Controlla che i landmark non siano tutti uguali
        let first = landmarks.first!
        let allSame = landmarks.allSatisfy { $0.x == first.x && $0.y == first.y && $0.z == first.z }
        if allSame {
            print("[DEBUGGING] ATTENZIONE: Tutti i landmark hanno gli stessi valori")
            return false
        }
        
        return true
    }
}
extension Array where Element == Double {
    func elements(at indices: [Int]) -> [Double] {
        var result = [Double](repeating: 0.0, count: indices.count)
        for (i, index) in indices.enumerated() {
            if index < self.count {
                result[i] = self[index]
            }
        }
        return result
    }
    func chunked(into size: Int) -> [[Element]] {
            return stride(from: 0, to: count, by: size).map {
                Array(self[$0 ..< Swift.min($0 + size, count)])
            }
        }
}
