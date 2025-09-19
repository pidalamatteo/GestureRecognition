// FeatureSelector.swift
import Foundation
import Accelerate

/// Strategy protocol for different feature selection methods
protocol FeatureSelectionStrategy {
    func selectFeatures(from features: [[Double]]) -> [Int]
}

/// Correlation-based feature selection strategy
struct CorrelationBasedStrategy: FeatureSelectionStrategy {
    let threshold: Double
    
    func selectFeatures(from features: [[Double]]) -> [Int] {
        guard !features.isEmpty, !features[0].isEmpty else { return [] }
        
        let featureCount = features[0].count
        var featuresToKeep: [Int] = []
        
        // Calculate correlation matrix using Accelerate
        let correlationMatrix = calculateCorrelationMatrix(features: features)
        
        // Select uncorrelated features
        for i in 0..<featureCount {
            var highlyCorrelated = false
            for j in featuresToKeep {
                if abs(correlationMatrix[i][j]) > threshold {
                    highlyCorrelated = true
                    break
                }
            }
            
            if !highlyCorrelated {
                featuresToKeep.append(i)
            }
        }
        
        return featuresToKeep
    }
    
    private func calculateCorrelationMatrix(features: [[Double]]) -> [[Double]] {
        // Implementation using Accelerate for better performance
        let featureCount = features[0].count
        var matrix: [[Double]] = Array(repeating: Array(repeating: 0.0, count: featureCount), count: featureCount)
        
        // Convert to column-major format for Accelerate
        var columnMajorData = [Double]()
        for i in 0..<featureCount {
            columnMajorData.append(contentsOf: features.map { $0[i] })
        }
        
        // Calculate correlations using Accelerate
        for i in 0..<featureCount {
            for j in i..<featureCount {
                if i == j {
                    matrix[i][j] = 1.0
                } else {
                    let x = Array(columnMajorData[i*features.count..<(i+1)*features.count])
                    let y = Array(columnMajorData[j*features.count..<(j+1)*features.count])
                    matrix[i][j] = calculatePearsonCorrelation(x, y)
                    matrix[j][i] = matrix[i][j]
                }
            }
        }
        
        return matrix
    }
    
    private func calculatePearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        let n = Double(x.count)
        
        // Use Accelerate for faster calculations
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumX2 = 0.0
        var sumY2 = 0.0
        
        vDSP_sveD(x, 1, &sumX, vDSP_Length(x.count))
        vDSP_sveD(y, 1, &sumY, vDSP_Length(y.count))
        
        vDSP_dotprD(x, 1, y, 1, &sumXY, vDSP_Length(x.count))
        
        vDSP_svesqD(x, 1, &sumX2, vDSP_Length(x.count))
        vDSP_svesqD(y, 1, &sumY2, vDSP_Length(y.count))
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        return denominator != 0 ? numerator / denominator : 0
    }
}

/// Variance-based feature selection strategy
struct VarianceBasedStrategy: FeatureSelectionStrategy {
    let threshold: Double
    
    func selectFeatures(from features: [[Double]]) -> [Int] {
        guard !features.isEmpty else { return [] }
        
        let featureCount = features[0].count
        var featuresToKeep: [Int] = []
        
        for i in 0..<featureCount {
            let values = features.map { $0[i] }
            let variance = calculateVariance(values)
            
            if variance > threshold {
                featuresToKeep.append(i)
            }
        }
        
        return featuresToKeep
    }
    
    private func calculateVariance(_ values: [Double]) -> Double {
        let count = Double(values.count)
        var mean = 0.0
        vDSP_meanvD(values, 1, &mean, vDSP_Length(values.count))
        
        var squaredDifferences = [Double](repeating: 0.0, count: values.count)
        let meanArray = [Double](repeating: mean, count: values.count)
        
        vDSP_vsubD(meanArray, 1, values, 1, &squaredDifferences, 1, vDSP_Length(values.count))
        vDSP_vsqD(squaredDifferences, 1, &squaredDifferences, 1, vDSP_Length(values.count))
        
        var sumSquaredDifferences = 0.0
        vDSP_sveD(squaredDifferences, 1, &sumSquaredDifferences, vDSP_Length(values.count))
        
        return sumSquaredDifferences / count
    }
}

/// Feature selector using strategy pattern
struct FeatureSelector {
    private let strategy: FeatureSelectionStrategy
    
    init(strategy: FeatureSelectionStrategy) {
        self.strategy = strategy
    }
    
    func selectFeatures(from features: [[Double]]) -> [Int] {
        return strategy.selectFeatures(from: features)
    }
    
    // MARK: - Feature Importance Calculation
    
    static func calculateFeatureImportance(with samples: [LandmarkSample]) -> [String: Double] {
        // Implementation remains the same but could be optimized with Accelerate
        let featureNames = getFeatureNames()
        
        var featureColumns: [String: [Double]] = [:]
        var labelColumn: [String] = []
        
        for featureName in featureNames {
            featureColumns[featureName] = []
        }
        
        for sample in samples {
            let features = LandmarkUtils.prepareForTraining(sample: sample)
            
            for (index, value) in features.enumerated() {
                if index < featureNames.count {
                    let featureName = featureNames[index]
                    featureColumns[featureName]?.append(value)
                }
            }
            
            labelColumn.append(sample.label)
        }
        
        var featureImportance: [String: Double] = [:]
        
        for featureName in featureNames {
            if let featureValues = featureColumns[featureName] {
                let uniqueLabels = Array(Set(labelColumn))
                var numericalLabels: [Double] = []
                
                for label in labelColumn {
                    if let index = uniqueLabels.firstIndex(of: label) {
                        numericalLabels.append(Double(index))
                    }
                }
                
                let correlation = calculatePearsonCorrelation(featureValues, numericalLabels)
                featureImportance[featureName] = abs(correlation)
            }
        }
        
        return featureImportance
    }
    
    // MARK: - Feature Names
    
    static func getFeatureNames() -> [String] {
        // Implementation remains the same
        var names: [String] = []
        
        for i in 0..<21 {
            names.append(contentsOf: ["landmark_\(i)_x", "landmark_\(i)_y", "landmark_\(i)_z"])
        }
        
        let geometricNames = [
            "wrist_thumb_tip_dist", "wrist_index_tip_dist", "wrist_middle_tip_dist",
            "wrist_ring_tip_dist", "wrist_pinky_tip_dist", "thumb_joint_angle",
            "thumb_index_tip_dist", "index_middle_tip_dist", "middle_ring_tip_dist",
            "ring_pinky_tip_dist", "hand_spread_angle", "average_finger_length",
            "thumb_index_length_ratio"
        ]
        
        names.append(contentsOf: geometricNames)
        
        let fingerNames = ["thumb", "index", "middle", "ring", "pinky"]
        let fingerFeatureNames = ["direction_x", "direction_y", "angle", "length", "curvature"]
        
        for finger in fingerNames {
            for feature in fingerFeatureNames {
                names.append("\(finger)_\(feature)")
            }
        }
        
        let globalFeatureNames = [
            "hand_area", "hand_aspect_ratio", "center_offset_x", "center_offset_y"
        ]
        names.append(contentsOf: globalFeatureNames)
        
        return names
    }
    
    // Helper function for Pearson correlation
    private static func calculatePearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        let n = Double(x.count)
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumX2 = 0.0
        var sumY2 = 0.0
        
        vDSP_sveD(x, 1, &sumX, vDSP_Length(x.count))
        vDSP_sveD(y, 1, &sumY, vDSP_Length(y.count))
        
        vDSP_dotprD(x, 1, y, 1, &sumXY, vDSP_Length(x.count))
        
        vDSP_svesqD(x, 1, &sumX2, vDSP_Length(x.count))
        vDSP_svesqD(y, 1, &sumY2, vDSP_Length(y.count))
        
        let numerator = n * sumXY - sumX * sumY
        let denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
        
        return denominator != 0 ? numerator / denominator : 0
    }
}
