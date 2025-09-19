import Foundation
import UIKit

struct ExportManager {
    
    static func createJSONZip(from samples: [LandmarkSample], progressHandler: ((String) -> Void)? = nil) -> URL? {
        
        guard !samples.isEmpty else { return nil }
        
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let baseFolder = documentsURL.appendingPathComponent("CreateML_JSON")
        
        // Cleanup e creazione directory
        try? fileManager.removeItem(at: baseFolder)
        try? fileManager.createDirectory(at: baseFolder, withIntermediateDirectories: true)
        
        progressHandler?("Filtering geometric features...")
        let selectedIndices = performFeatureSelection(on: samples)
        
        progressHandler?("Saving selected features...")
        LandmarkUtils.saveSelectedFeatureIndices(selectedIndices)
        
        progressHandler?("Balancing classes...")
        let balancedSamples = balanceClasses(samples: samples)
        let groupedSamples = Dictionary(grouping: balancedSamples, by: { $0.label })
        let subsets = ["Training": 0.7, "Validation": 0.1, "Testing": 0.2]
        
        for (subsetName, ratio) in subsets {
            progressHandler?("Preparing JSON \(subsetName)...")
            let subsetFolder = baseFolder.appendingPathComponent(subsetName)
            try? fileManager.createDirectory(at: subsetFolder, withIntermediateDirectories: true)
            
            for (label, samples) in groupedSamples {
                
                let labelFolder = subsetFolder.appendingPathComponent(label)
                try? fileManager.createDirectory(at: labelFolder, withIntermediateDirectories: true)
                
                let samplesToSave = samples
                    .shuffled()
                    .prefix(Int(Double(samples.count) * ratio))
                
                for (index, sample) in samplesToSave.enumerated() {
                    progressHandler?("Writing \(subsetName) / \(label) \(index + 1)/\(samplesToSave.count)...")
                    let fullFeatures = LandmarkUtils.prepareForTraining(sample: sample)
                    let selectedFeatures = selectedIndices.map { fullFeatures[$0] }
                    
                    saveSampleAsJSON(features: selectedFeatures, label: label,
                                   index: index, folder: labelFolder)
                }
            }
        }
        
        progressHandler?("Saving metadates...")
        saveFeatureSelectionMetadata(selectedIndices, in: baseFolder)
        
        progressHandler?("Creation final ZIP...")
        return zipFolder(at: baseFolder)
    }
    
    // Nuovo metodo per eseguire il feature selection
    private static func performFeatureSelection(on samples: [LandmarkSample]) -> [Int] {
        // Estrai tutte le feature complete
        let allFeatures = samples.map { LandmarkUtils.prepareForTraining(sample: $0) }
        
        // 1. Rimuovi feature a bassa varianza usando la strategia
        let varianceStrategy = VarianceBasedStrategy(threshold: 0.005)
        let varianceSelector = FeatureSelector(strategy: varianceStrategy)
        let varianceIndices = varianceSelector.selectFeatures(from: allFeatures)
        
        // Filtra le feature mantenendo solo quelle con varianza sufficiente
        let varianceFilteredFeatures = allFeatures.map { featureArray in
            varianceIndices.map { featureArray[$0] }
        }
        
        // 2. Rimuovi feature altamente correlate usando la strategia
        let correlationStrategy = CorrelationBasedStrategy(threshold: 0.9)
        let correlationSelector = FeatureSelector(strategy: correlationStrategy)
        let correlationIndices = correlationSelector.selectFeatures(from: varianceFilteredFeatures)
        
        // Combina i due filtri: prima varianza, poi correlazione
        let combinedIndices = correlationIndices.map { varianceIndices[$0] }
        
        // 3. Calcola l'importanza delle feature
        let featureImportanceDict = FeatureSelector.calculateFeatureImportance(with: samples)
        let allFeatureNames = FeatureSelector.getFeatureNames()
        
        // Filtra l'importanza solo per le feature selezionate
        var filteredImportance: [(index: Int, importance: Double)] = []
        for index in combinedIndices {
            let featureName = allFeatureNames[index]
            if let importance = featureImportanceDict[featureName] {
                filteredImportance.append((index, importance))
            }
        }
        
        /*
        // Ordina per importanza e seleziona le top 30
        let topFeatureIndices = filteredImportance
            .sorted { $0.importance > $1.importance }
            .prefix(30)
            .map { $0.index }
        
        return topFeatureIndices
         */
        let topFeatureIndices = filteredImportance
            .sorted { $0.importance > $1.importance }
            .prefix(30)
            .map { $0.index }
        
        let validIndices = topFeatureIndices.filter { $0 < allFeatureNames.count }
            
            print("Selected \(validIndices.count) features:")
            for (i, index) in validIndices.enumerated() {
                let featureName = allFeatureNames[index]
                if let importance = featureImportanceDict[featureName] {
                    print("\(i+1). \(featureName): \(importance)")
                }
            }
            
            return validIndices
    }
    
    // Metodo per salvare i metadati del feature selection
    private static func saveFeatureSelectionMetadata(_ indices: [Int], in folder: URL) {
        let allFeatureNames = FeatureSelector.getFeatureNames()
        let selectedFeatureNames = indices.map { allFeatureNames[$0] }
        
        let metadata: [String: Any] = [
            "selected_feature_indices": indices,
            "selected_feature_names": selectedFeatureNames,
            "total_features": allFeatureNames.count,
            "selected_features_count": indices.count,
            "selection_timestamp": Date().timeIntervalSince1970
        ]
        
        let metadataURL = folder.appendingPathComponent("feature_selection_metadata.json")
        if let jsonData = try? JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted) {
            try? jsonData.write(to: metadataURL)
        }
    }
    
    private static func saveSampleAsJSON(features: [Double], label: String,
                                       index: Int, folder: URL) {
        let jsonDict: [String: Any] = [
            "label": label,
            "features": features
        ]
        
        let fileURL = folder.appendingPathComponent("\(label)_\(index).json")
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: jsonDict,
                                                    options: [.prettyPrinted]) {
            try? jsonData.write(to: fileURL)
        }
    }
    
    private static func zipFolder(at folderURL: URL) -> URL? {
        let fileManager = FileManager.default
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let zipURL = documentsURL.appendingPathComponent("Gestures_JSON.zip")
        
        try? fileManager.removeItem(at: zipURL)
        
        let coordinator = NSFileCoordinator()
        var error: NSError?
        var resultURL: URL?
        
        coordinator.coordinate(readingItemAt: folderURL,
                             options: .forUploading,
                             error: &error) { zippedURL in
            do {
                try fileManager.copyItem(at: zippedURL, to: zipURL)
                resultURL = zipURL
            } catch {
                print("Errore creazione ZIP: \(error)")
            }
        }
        
        return resultURL
    }
    
    private static func balanceClasses(samples: [LandmarkSample]) -> [LandmarkSample] {
        let classGroups = Dictionary(grouping: samples, by: { $0.label })
        let maxCount = classGroups.values.map { $0.count }.max() ?? 0
        
        return classGroups.flatMap { label, samples in
            var balanced = samples
            while balanced.count < maxCount {
                if let randomSample = samples.randomElement() {
                    balanced.append(LandmarkUtils.augment(sample: randomSample))
                }
            }
            return balanced
        }
    }
}
