/*
func testWithKnownGestures() {
    // Carica alcuni campioni conosciuti dal tuo dataset di training
    let knownSamples = loadKnownSamples() // Implementa questa funzione
    
    for sample in knownSamples {
        let features = LandmarkUtils.prepareForPrediction(from: sample.landmarks)
        let (prediction, confidence) = predictWithDebug(features: features)
        
        print("Expected: \(sample.label), Got: \(prediction), Confidence: \(confidence)")
        
        // Se la prediction è sbagliata, investiga
        if prediction != sample.label {
            print("PREDICTION ERRATA - Investigare!")
            // Potresti voler salvare i feature vector per analisi successive
        }
    }
}
*/
