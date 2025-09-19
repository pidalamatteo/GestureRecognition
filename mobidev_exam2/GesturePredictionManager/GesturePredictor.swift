// GesturePredictor.swift
import Foundation
import CoreML

/// Handles gesture prediction using a CoreML model with confidence threshold management
class GesturePredictor: ObservableObject {
    private var model: GestureClassifier?
    private let thresholdManager = ConfidenceThresholdManager()
    private var metrics: MetricsStruct?
    
    // Cache for recent predictions with time-based expiration
    private var predictionHistory: [(label: String, confidence: Double, timestamp: Date)] = []
    private let historyTimeWindow: TimeInterval = 1.5
    private let maxHistoryCount = 50 // Prevent memory leaks
    
    // Configuration temporal smoothing
    @Published var config = SmoothingConfig()

     struct SmoothingConfig {
        var timeWindow: TimeInterval = 1.5
        var minConfidenceThreshold: Double = 0.5
        var minStableFrames: Int = 2
        var requiredConsensusRatio: Double = 0.5

         static func defaultValue() -> SmoothingConfig {
             return SmoothingConfig(
                 timeWindow: 1.5,
                 minConfidenceThreshold: 0.5,
                 minStableFrames: 2,
                 requiredConsensusRatio: 0.5
             )
         }
    }
    
    // Dedicated code for prediction
    private var lastPrediction: (label: String, confidence: Double)?
    private let predictionQueue = DispatchQueue(label: "com.yourapp.predictionQueue", qos: .userInitiated, attributes: .concurrent)
    
    init() {
        loadModel()
    }
    
    func predictFromFeatures(_ features: [Double], completion: @escaping ((label: String, confidence: Double)?) -> Void) {
           // Esegui in background per non bloccare il thread UI
       predictionQueue.async {
           let finalFeatures: [Double]
           
           if features.count == 30 {
               finalFeatures = features
           } else {
               guard let selectedIndices = LandmarkUtils.selectedFeatureIndices else {
                   DispatchQueue.main.async { completion(nil) }
                   return
               }
               
               guard selectedIndices.allSatisfy({ $0 < features.count }) else {
                   DispatchQueue.main.async { completion(nil) }
                   return
               }
               
               finalFeatures = selectedIndices.map { features[$0] }
           }
           
           let result = self.predictWithFeatures(finalFeatures)
           DispatchQueue.main.async { completion(result) }
       }
    }

    /// Internal prediction method with feature validation
    private func predictWithFeatures(_ features: [Double]) -> (label: String, confidence: Double)? {
       guard let model = model, features.count == 30 else {
           return nil
       }
       
       do {
           let input = try createModelInput(from: features)
           let prediction = try model.prediction(input: input)
           
           guard let (label, confidence) = prediction.labelProbability.max(by: { $0.value < $1.value }) else {
               return nil
           }
           
           lastPrediction = (label, confidence)
           
           return thresholdManager.shouldAcceptPrediction(label, confidence: confidence) ?
                  (label, confidence) : nil
       } catch {
           print("[DEBUGGING] Error during prediction: \(error.localizedDescription)")
           return nil
       }
   }
    
    /// Makes a prediction with temporal smoothing
    func predictWithTemporalSmoothing(from features: [Double], completion: @escaping ((label: String, confidence: Double)?) -> Void) {
        predictionQueue.async {
            guard let currentPrediction = self.predictWithFeatures(features) else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            
            let now = Date()
            self.predictionHistory.append((currentPrediction.label, currentPrediction.confidence, now))
            self.cleanupHistory(currentTime: now)
            
            // Se la confidence è molto alta, restituisci immediatamente
            if currentPrediction.confidence > 0.85 {
                DispatchQueue.main.async { completion(currentPrediction) }
                return
            }
            
            let result = self.applyTemporalFilter()
            DispatchQueue.main.async { completion(result) }
        }
    }
    
    /// Removes old predictions from history
    private func cleanupHistory(currentTime: Date) {
        // Remove predictions older than the time window
        predictionHistory = predictionHistory.filter {
            currentTime.timeIntervalSince($0.timestamp) <= config.timeWindow
        }
        
        // Also limit the total count to prevent memory issues
        if predictionHistory.count > maxHistoryCount {
            predictionHistory.removeFirst(predictionHistory.count - maxHistoryCount)
        }
    }
    
    /// Applies temporal filtering to smooth predictions
    private func applyTemporalFilter() -> (label: String, confidence: Double)? {
       guard predictionHistory.count >= config.minStableFrames else {
           return predictionHistory.last.map { ($0.label, $0.confidence) }
       }
       
       var classConfidences: [String: (total: Double, count: Int)] = [:]
       var totalConfidence: Double = 0
       var totalCount = 0
       
       for (label, confidence, _) in predictionHistory {
           classConfidences[label, default: (0, 0)].total += confidence
           classConfidences[label]?.count += 1
           totalConfidence += confidence
           totalCount += 1
       }
       
       // Calcola la confidence media totale
       let averageTotalConfidence = totalConfidence / Double(totalCount)
       
       // Se la confidence media totale è bassa, restituisci nil
       guard averageTotalConfidence >= config.minConfidenceThreshold else {
           return nil
       }
       
       // Trova la label con la confidence media più alta
       guard let bestClass = classConfidences.max(by: {
           ($0.value.total / Double($0.value.count)) < ($1.value.total / Double($1.value.count))
       }) else {
           return nil
       }
       
       let averageConfidence = bestClass.value.total / Double(bestClass.value.count)
       let bestClassRatio = Double(bestClass.value.count) / Double(totalCount)
       
       return (bestClassRatio >= config.requiredConsensusRatio &&
               averageConfidence >= config.minConfidenceThreshold) ?
               (bestClass.key, averageConfidence) : nil
   }
    /// Alternative: simple voting mode with threshold
    private func applySimpleVoting() -> (label: String, confidence: Double)? {
        var voteCounts: [String: Int] = [:]
        
        // Count votes only for predictions above the threshold
        for (label, confidence, _) in predictionHistory where confidence >= config.minConfidenceThreshold {
            voteCounts[label, default: 0] += 1
        }
        
        guard let bestLabel = voteCounts.max(by: { $0.value < $1.value })?.key else {
            return nil
        }
        
        // Calculate average confidence for the winning label
        let confidences = predictionHistory
            .filter { $0.label == bestLabel && $0.confidence >= config.minConfidenceThreshold }
            .map { $0.confidence }
        
        guard !confidences.isEmpty else {
            return nil
        }
        
        let averageConfidence = confidences.reduce(0, +) / Double(confidences.count)
        return (bestLabel, averageConfidence)
    }
    
    /// Reset the prediction history
    func resetTemporalHistory() {
        predictionHistory.removeAll()
    }
    
    /// Gets the current history state for debugging
    func getHistoryState() -> String {
        var state = "History count: \(predictionHistory.count)\n"
        
        let grouped = Dictionary(grouping: predictionHistory, by: { $0.label })
        for (label, predictions) in grouped {
            let avgConf = predictions.map { $0.confidence }.reduce(0, +) / Double(predictions.count)
            state += "\(label): \(predictions.count) frames, avg conf: \(String(format: "%.2f", avgConf))\n"
        }
        
        return state
    }
    
    /// Loads metrics from a file and calculates optimal thresholds
    @discardableResult
    func loadMetrics(from url: URL) -> Bool {
        do {
            let data = try Data(contentsOf: url)
            let metrics = try JSONDecoder().decode(MetricsStruct.self, from: data)
            self.metrics = metrics
            thresholdManager.calculateOptimalThresholds(from: metrics)
            print("Metrics loaded successfully")
            return true
        } catch {
            print("Error loading metrics: \(error.localizedDescription)")
            return false
        }
    }
    
    /// Returns all current thresholds
    func getAllThresholds() -> [String: Double] {
        return thresholdManager.getAllThresholds()
    }
    
    /// Gets the last prediction made by the model
    func getLastPrediction() -> (label: String, confidence: Double)? {
        return lastPrediction
    }
    
    // MARK: - Model Input Creation
    private func createModelInput(from features: [Double]) throws -> GestureClassifierInput {
        // Ensure we have exactly 30 features
        guard features.count == 30 else {
            throw PredictionError.invalidFeatureCount(expected: 30, actual: features.count)
        }
        
        return GestureClassifierInput(
            feature_0: features[0], feature_1: features[1], feature_2: features[2],
            feature_3: features[3], feature_4: features[4], feature_5: features[5],
            feature_6: features[6], feature_7: features[7], feature_8: features[8],
            feature_9: features[9], feature_10: features[10], feature_11: features[11],
            feature_12: features[12], feature_13: features[13], feature_14: features[14],
            feature_15: features[15], feature_16: features[16], feature_17: features[17],
            feature_18: features[18], feature_19: features[19], feature_20: features[20],
            feature_21: features[21], feature_22: features[22], feature_23: features[23],
            feature_24: features[24], feature_25: features[25], feature_26: features[26],
            feature_27: features[27], feature_28: features[28], feature_29: features[29]
        )
    }
    
    private func loadModel() {
        do {
            model = try GestureClassifier(configuration: MLModelConfiguration())
            print("Gesture model loaded successfully")
        } catch {
            print("Error loading gesture model: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Debug Functions
    
    /// Makes prediction with detailed debug information
    func predictWithDebug(features: [Double]) -> (label: String, confidence: Double)? {
        guard let model = model else {
            print("[DEBUGGING] Error: Model not loaded")
            return nil
        }
        
        do {
            let input = try createModelInput(from: features)
            let prediction = try model.prediction(input: input)
            
            // Print all probabilities
            print("[DEBUGGING] All class probabilities: \(prediction.labelProbability)")
            
            // Find the prediction with the highest probability
            guard let (label, confidence) = prediction.labelProbability.max(by: { $0.value < $1.value }) else {
                print("[DEBUGGING] Error: Could not get prediction probabilities")
                return nil
            }
            
            print("[DEBUGGING] Raw prediction: \(label)")
            print("[DEBUGGING] Confidence: \(confidence)")
            
            // Find the second best prediction
            let sorted = prediction.labelProbability.sorted(by: { $0.value > $1.value })
            if sorted.count >= 2 {
                print("[DEBUGGING] Top prediction: \(sorted[0].key) = \(sorted[0].value)")
                print("[DEBUGGING] Second prediction: \(sorted[1].key) = \(sorted[1].value)")
            }
            
            return (label, confidence)
        } catch {
            print("[DEBUGGING] Error during prediction: \(error.localizedDescription)")
            return nil
        }
    }
    
    /// Gets all probabilities from the model
    private func getAllProbabilities(from features: [Double]) -> [String: Double]? {
        guard let model = model else {
            print("Error: Model not loaded")
            return nil
        }
        
        do {
            let input = try createModelInput(from: features)
            let prediction = try model.prediction(input: input)
            return prediction.labelProbability
        } catch {
            print("Error getting probabilities: \(error.localizedDescription)")
            return nil
        }
    }
}

enum PredictionError: LocalizedError {
    case invalidFeatureCount(expected: Int, actual: Int)
    
    var errorDescription: String? {
        switch self {
        case .invalidFeatureCount(let expected, let actual):
            return "Expected \(expected) features, but got \(actual)"
        }
    }
}
