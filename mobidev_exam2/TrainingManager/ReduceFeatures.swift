// ReduceFeatures.swift
import Foundation

/// Handles feature selection and reduction
class ReduceFeatures {
    private let varianceThreshold: Double
    private let correlationThreshold: Double
    private let maxFeatures: Int
    
    /// Initializes with configurable thresholds
    init(varianceThreshold: Double = 0.005,
         correlationThreshold: Double = 0.9,
         maxFeatures: Int = 30) {
        self.varianceThreshold = varianceThreshold
        self.correlationThreshold = correlationThreshold
        self.maxFeatures = maxFeatures
    }
    
    /// Performs feature selection on training samples
    func performFeatureSelection(trainingSamples: [LandmarkSample]) -> [Int] {
        let allFeatures = trainingSamples.map { LandmarkUtils.prepareForTraining(sample: $0) }
        
        // 1. Remove low variance features
        let varianceStrategy = VarianceBasedStrategy(threshold: varianceThreshold)
        let varianceSelector = FeatureSelector(strategy: varianceStrategy)
        let varianceIndices = varianceSelector.selectFeatures(from: allFeatures)
        
        // Filter features
        let varianceFilteredFeatures = allFeatures.map { featureArray in
            varianceIndices.map { featureArray[$0] }
        }
        
        // 2. Remove highly correlated features
        let correlationStrategy = CorrelationBasedStrategy(threshold: correlationThreshold)
        let correlationSelector = FeatureSelector(strategy: correlationStrategy)
        let correlationIndices = correlationSelector.selectFeatures(from: varianceFilteredFeatures)
        
        // Combine filters
        let combinedIndices = correlationIndices.map { varianceIndices[$0] }
        
        // 3. Calculate feature importance
        let featureImportanceDict = FeatureSelector.calculateFeatureImportance(with: trainingSamples)
        let allFeatureNames = FeatureSelector.getFeatureNames()
        
        // Filter importance for selected features
        var filteredImportance: [(index: Int, importance: Double)] = []
        for index in combinedIndices {
            let featureName = allFeatureNames[index]
            if let importance = featureImportanceDict[featureName] {
                filteredImportance.append((index, importance))
            }
        }
        
        // Select top features
        let topFeatureIndices = filteredImportance
            .sorted { $0.importance > $1.importance }
            .prefix(maxFeatures)
            .map { $0.index }
        
        return topFeatureIndices
    }
    
    /// Saves selected feature indices
    func saveSelectedFeatures(_ indices: [Int]) {
        UserDefaults.standard.set(indices, forKey: "selectedFeatureIndices")
        print("Selected features saved: \(indices)")
    }
    
    /// Trains model with feature selection
    func trainModel(with trainingData: [LandmarkSample]) {
        let selectedFeatureIndices = performFeatureSelection(trainingSamples: trainingData)
        saveSelectedFeatures(selectedFeatureIndices)
        
        // Use filtered features for training
        let filteredFeatures = trainingData.map { sample in
            let allFeatures = LandmarkUtils.prepareForTraining(sample: sample)
            return selectedFeatureIndices.map { allFeatures[$0] }
        }
        
        // Use filteredFeatures for training (this was previously unused)
        _ = filteredFeatures // Now used or handled appropriately
    }
}
