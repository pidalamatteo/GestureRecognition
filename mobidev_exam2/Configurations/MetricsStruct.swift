import Foundation

import Foundation

struct MetricsStruct: Codable {
    let testing: MetricsData
    let validation: MetricsData
    
    struct MetricsData: Codable {
        let accuracy: Double
        let balancedAccuracy: Double
        let classMetrics: [String: ClassMetric]
        let confusionMatrix: [String: [String: Int]]
        
        enum CodingKeys: String, CodingKey {
            case accuracy
            case balancedAccuracy = "balanced_accuracy"
            case classMetrics = "class_metrics"
            case confusionMatrix = "confusion_matrix"
        }
    }
    
    struct ClassMetric: Codable {
        let precision: Double
        let recall: Double
        let f1: Double
    }
}

