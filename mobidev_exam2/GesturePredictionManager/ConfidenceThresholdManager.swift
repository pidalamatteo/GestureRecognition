import Foundation
import CoreML

class ConfidenceThresholdManager {
    private var classThresholds: [String: Double] = [:]
    private let defaultThreshold: Double = 0.6
    
    func calculateOptimalThresholds(from metrics: MetricsStruct) {
        for (className, metric) in metrics.testing.classMetrics {
            // Pulisci il nome della classe
            let cleanClassName = className
                .replacingOccurrences(of: "Optional(\"", with: "")
                .replacingOccurrences(of: "\")", with: "")
            
            // Calcola la soglia ottimale
            let threshold = calculateThresholdForClass(
                precision: metric.precision,
                recall: metric.recall,
                f1: metric.f1
            )
            
            classThresholds[cleanClassName] = threshold
        }
    }
     
    
    private func calculateThresholdForClass(precision: Double, recall: Double, f1: Double) -> Double {
        // Strategia: classi con precision più bassa necessitano di threshold più alti
        let precisionWeight = 0.7
        let recallWeight = 0.3
        
        let baseThreshold = 0.5
        let precisionAdjustment = (1 - precision) * precisionWeight
        let recallAdjustment = (1 - recall) * recallWeight
        
        let calculatedThreshold = min(0.95, max(0.3, baseThreshold + precisionAdjustment - recallAdjustment))
        
        print("Soglia per classe: precision=\(precision), recall=\(recall) → threshold=\(calculatedThreshold)")
        
        return calculatedThreshold
    }
    
    func getThreshold(for className: String) -> Double {
        return classThresholds[className] ?? defaultThreshold
    }
    
    func shouldAcceptPrediction(_ prediction: String, confidence: Double) -> Bool {
        let threshold = getThreshold(for: prediction)
        return confidence >= threshold
    }
    
    func getAllThresholds() -> [String: Double] {
        return classThresholds
    }
}
