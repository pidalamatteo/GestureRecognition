import SwiftUI
import AVFoundation
import MediaPipeTasksVision
import CoreML

class HandLandmarkerViewModel: NSObject, ObservableObject {
    
    // MARK: - Configurazioni
    @ObservedObject var config = DefaultCostants()
    
    // MARK: - Risultati
    @Published var detectedHands: [[NormalizedLandmark]] = []
    @Published var inferenceTime: Double = 0.0
    @Published var frameInterval: Double = 0.0
    @Published var currentImageSize: CGSize? = nil
    
    // MARK: - Gestione Gesture Prediction
    private var gesturePredictor = GesturePredictor()
    @Published var confidenceThresholds: [String: Double] = [:]
    @Published var gestureRecognized: String = "Unknown"
    @Published var isGestureRecognized: Bool = false
    @Published var predictionConfidence: Double = 0.0
    
    // MARK: - Gestione Registrazione
    @Published var gestureLabelToRegister: String = ""
    @Published var isRecordingGesture = false
    @Published var totalRecordingTime: Double = 0 // ms
    @Published var avgPresence: Float = 0.0
    
    // MARK: - Gestione Thread-Safe delle samples
    private let samplesQueue = DispatchQueue(label: "com.yourapp.samplesQueue", attributes: .concurrent)
    private var internalRecordedSamples: [LandmarkSample] = []
    @Published var recordedSamples: [LandmarkSample] = [] {
        didSet {
            self.saveSamplesToFile()
        }
    }
    
    private var startTime: CFTimeInterval = 0
    private var lastFrameTimestamp: CFTimeInterval? = nil
    private var handLandmarkerService: HandLandmarkerServiceLiveStream?
    private var recordingStartTime: CFTimeInterval?
    private var recordingFrameCounter: Int = 0
    
    private let saveFileName = "gestures.json"
    private var saveURL: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent(saveFileName)
    }
    
    // MARK: - Inizializzazione
    override init() {
        super.init()
        setupLandmarker()
        loadSamples()
        loadModelMetrics()
        LandmarkUtils.debugFileLocations()
        loadFeatureIndices()
        
        // Esegui test di consistenza dopo un breve delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.testPredictionConsistency()
        }
    }
    
    // MARK: - Setup Landmarker
    private func setupLandmarker() {
        guard let modelPath = config.modelPath else {
            fatalError("Modello hand_landmarker.task non trovato nel bundle")
        }
        
        handLandmarkerService = HandLandmarkerServiceLiveStream(
            modelPath: modelPath,
            numHands: config.numHands,
            minHandDetectionConfidence: config.minHandDetectionConfidence,
            minHandPresenceConfidence: config.minHandPresenceConfidence,
            minTrackingConfidence: config.minTrackingConfidence
        )
        
        handLandmarkerService?.delegate = self
    }
    
    // MARK: - Caricamento metriche del modello e feature selection
    private func loadModelMetrics() {
        if let metricsURL = Bundle.main.url(forResource: "model_metrics", withExtension: "json") {
            let success = gesturePredictor.loadMetrics(from: metricsURL)
            if success {
                confidenceThresholds = gesturePredictor.getAllThresholds()
                print("Metriche del modello caricate con successo")
            }
        }
    }
    
    private func loadFeatureIndices() {
        // Carica gli indici delle feature selezionate
        _ = LandmarkUtils.loadSelectedFeatureIndices()
        print("Indici delle feature selezionate caricati")
    }
    
    // MARK: - Aggiornamento opzioni
    func updateOptions() {
        guard let modelPath = config.modelPath else { return }
        
        handLandmarkerService = HandLandmarkerServiceLiveStream(
            modelPath: modelPath,
            numHands: config.numHands,
            minHandDetectionConfidence: config.minHandDetectionConfidence,
            minHandPresenceConfidence: config.minHandPresenceConfidence,
            minTrackingConfidence: config.minTrackingConfidence
        )
        
        handLandmarkerService?.delegate = self
        print("HandLandmarker aggiornato: numHands=\(config.numHands), minDetection=\(config.minHandDetectionConfidence)")
    }
    
    // MARK: - Processamento frame
    func processFrame(_ sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation) {
        startTime = CACurrentMediaTime()
        handLandmarkerService?.detectAsync(sampleBuffer: sampleBuffer, orientation: orientation)
    }
    
    // MARK: - Gestione Registrazione Campioni
    func startRecordingGesture(label: String) {
        gestureLabelToRegister = label
        recordingFrameCounter = 0
        isRecordingGesture = true
        totalRecordingTime = 0
        recordingStartTime = CACurrentMediaTime()
        print("Inizio registrazione gesto [\(label)]")
        
        // Resetta la history delle prediction quando inizi a registrare
        gesturePredictor.resetTemporalHistory()
    }
    
    func stopRecordingGesture() {
        isRecordingGesture = false
        if let start = recordingStartTime {
            totalRecordingTime = (CACurrentMediaTime() - start)
        }
        printRecordedSamples()
    }
    
    func printRecordedSamples() {
        print("--- Gesti registrati ---")
        for (index, sample) in getSamples().enumerated() {
            print("Sample #\(index + 1) | Label: \(sample.label) | Landmarks: \(sample.landmarks.count)")
        }
        print("Totale campioni registrati: \(getSamples().count)")
    }
    
    // MARK: - Metodi Thread-Safe per recordedSamples
    private func addSample(_ sample: LandmarkSample) {
        samplesQueue.async(flags: .barrier) {
            self.internalRecordedSamples.append(sample)
            DispatchQueue.main.async {
                self.recordedSamples = self.internalRecordedSamples
            }
        }
    }
    
    func getSamples() -> [LandmarkSample] {
        return samplesQueue.sync {
            return self.internalRecordedSamples
        }
    }
    
    private func clearSamples() {
        samplesQueue.async(flags: .barrier) {
            self.internalRecordedSamples.removeAll()
            DispatchQueue.main.async {
                self.recordedSamples = []
            }
        }
    }
    
    private func saveSamplesToFile() {
        do {
            let data = try JSONEncoder().encode(self.recordedSamples)
            try data.write(to: saveURL)
            print("Gestures salvati in \(saveURL)")
        } catch {
            print("Errore salvataggio gestures: \(error)")
        }
    }
    
    private func loadSamples() {
        do {
            let data = try Data(contentsOf: saveURL)
            let samples = try JSONDecoder().decode([LandmarkSample].self, from: data)
            samplesQueue.async(flags: .barrier) {
                self.internalRecordedSamples = samples
                DispatchQueue.main.async {
                    self.recordedSamples = samples
                }
            }
            print("Gestures caricati (\(samples.count))")
        } catch {
            print("Nessun file gestures trovato o errore: \(error)")
        }
    }
    
    func clearSavedSamples() {
        do {
            try FileManager.default.removeItem(at: saveURL)
            clearSamples()
            print("File gestures eliminato")
        } catch {
            print("Errore eliminazione file: \(error)")
        }
    }
    
    func removeSamples(for label: String) {
        samplesQueue.async(flags: .barrier) {
            self.internalRecordedSamples.removeAll { $0.label == label }
            DispatchQueue.main.async {
                self.recordedSamples = self.internalRecordedSamples
            }
        }
    }
    /*
    
    // MARK: - Predizione gesto (usando feature selection)
    func predictGesture(from landmarks: [LandmarkPoint]) -> (String, Double)? {
        // Valida i landmark prima della prediction
        guard LandmarkUtils.validateLandmarks(landmarks) else {
            print("[DEBUGGING] Landmark non validi, prediction annullata")
            return nil
        }
        
        // Usa prepareForPrediction che applica il feature selection
        let features = LandmarkUtils.prepareForPrediction(from: landmarks)
        
        // DEBUG: stampa le feature estratte
        print("[DEBUGGING] Features estratte: \(features.count)")
        print("[DEBUGGING] Prime 5 feature: \(features.prefix(5))")
        
        if let result = gesturePredictor.predictWithTemporalSmoothing(from: features) {
            return (result.label, result.confidence)
        } else {
            // Se la prediction fallisce, prova a usare il metodo di debug per avere più informazioni
            if let debugResult = gesturePredictor.predictWithDebug(features: features) {
                print("[DEBUGGING] Predizione di debug: \(debugResult.label) con confidenza \(debugResult.confidence)")
            }
            return nil
        }
    }
     */
    
    // MARK: - Utility
    func boundingBox(for landmarks: [NormalizedLandmark], originalSize: CGSize, viewSize: CGSize, isBackCamera: Bool) -> CGRect? {
        guard !landmarks.isEmpty else { return nil }

        var xs = landmarks.map { CGFloat($0.y) * viewSize.width }
        let ys = landmarks.map { CGFloat($0.x) * viewSize.height }
       
        if isBackCamera {
            xs = landmarks.map { viewSize.width - (CGFloat($0.y) * viewSize.width) }
        }
        let minX = xs.min() ?? 0
        let maxX = xs.max() ?? 0
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? 0
        
        let padding: CGFloat = 20
        return CGRect(
            x: CGFloat(minX) - padding,
            y: CGFloat(minY) - padding,
            width: CGFloat(maxX - minX) + 2*padding,
            height: CGFloat(maxY - minY) + 2*padding
        )
    }
    
    func getSavedFileURL() -> URL? {
        let url = saveURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        } else {
            return nil
        }
    }
    
    func testPredictionConsistency() {
        // Crea landmark di test
        let testLandmarks = (0..<21).map { i in
            LandmarkPoint(
                x: Float(0.1 * Double(i)),
                y: Float(0.2 * Double(i)),
                z: Float(0.3 * Double(i))
            )
        }
        
        /*
        // Testa la prediction
        if let result = predictGesture(from: testLandmarks) {
            print("[DEBUGGING] Test prediction: \(result.0) - \(result.1)")
        } else {
            print("[DEBUGGING] Test prediction fallita")
        }
        */
        // Verifica la consistenza della normalizzazione
        LandmarkUtils.verifyNormalizationConsistency()
        
        // Verifica gli indici delle feature
        LandmarkUtils.verifyFeatureIndices()
    }
    
    func testConsistencyAfterFix() {
        let testLandmarks = (0..<21).map { i in
            LandmarkPoint(x: Float(0.1 * Double(i)), y: Float(0.2 * Double(i)), z: Float(0.3 * Double(i)))
        }
        
        // Testa la normalizzazione
        LandmarkUtils.verifyNormalizationConsistency()
        
        // Testa l'intera pipeline
        let features = LandmarkUtils.prepareForPrediction(from: testLandmarks)
        print("[DEBUGGING] Features after fix: \(features.prefix(5))...")
        
        /*
        if let result = predictGesture(from: testLandmarks) {
            print("[DEBUGGING] Prediction after fix: \(result.0) - \(result.1)")
        }
         */
    }
    
    // Reset della history quando necessario
    func resetPredictionHistory() {
        gesturePredictor.resetTemporalHistory()
    }
    
    // Esegui test di debug
    func runDebugTests() {
        testPredictionConsistency()
        
        // Verifica gli indici delle feature selezionate
        if let indices = LandmarkUtils.selectedFeatureIndices {
            print("[DEBUGGING] Indici feature selezionate: \(indices)")
            print("[DEBUGGING] Numero di indici: \(indices.count)")
        } else {
            print("[DEBUGGING] Nessun indice di feature selezionate disponibile")
        }
        
        // Verifica le thresholds
        print("[DEBUGGING] Thresholds: \(confidenceThresholds)")
        
        // Stampa lo stato della history
        print("[DEBUGGING] Stato history: \(gesturePredictor.getHistoryState())")
    }
}

// MARK: - Delegate
extension HandLandmarkerViewModel: HandLandmarkerServiceLiveStreamDelegate {
    func didDetectHands(_ result: HandLandmarkerResult) {
        let now = CACurrentMediaTime()
        let elapsed = (now - startTime) * 1000.0
        var delta: Double = 0
        if let last = lastFrameTimestamp {
            delta = (now - last) * 1000.0
        }
        lastFrameTimestamp = now
        
        guard let hand = result.landmarks.first else {
            gesturePredictor.resetTemporalHistory()
            
            DispatchQueue.main.async {
                self.updateUI(elapsed: elapsed, delta: delta, landmarks: result.landmarks,
                              gesture: "No hand", confidence: 0.0, isRecognized: false)
            }
            return
        }
        
        let points = hand.map { LandmarkPoint(from: $0) }
        
        // --- REGISTRAZIONE ---
        if self.isRecordingGesture {
            self.handleRecording(points: points)
        } else {
            // --- PREDICTION (con feature selection) ---
            self.handlePrediction(points: points)
        }
        
        // --- AGGIORNAMENTO UI GENERALE ---
        DispatchQueue.main.async {
            self.inferenceTime = elapsed
            self.frameInterval = delta
            self.detectedHands = result.landmarks
            
            if self.isRecordingGesture, let start = self.recordingStartTime {
                self.totalRecordingTime = now - start
            }
        }
    }
    private func handleRecording(points: [LandmarkPoint]) {
        recordingFrameCounter += 1
        
        if recordingFrameCounter % 2 == 0,
           shouldSaveSample(newPoints: points,
                           lastSaved: getSamples().last?.landmarks,
                           config: config,
                           distanceThreshold: config.minFrameDistance,
                           minVisibleLandmarks: 5) {
            
            let sample = LandmarkSample(label: gestureLabelToRegister, landmarks: points)
            addSample(sample)
            
            DispatchQueue.main.async {
                self.isGestureRecognized = true
                self.gestureRecognized = "Saved"
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.isGestureRecognized = false
                    self.gestureRecognized = "Recording..."
                }
            }
        }
    }
    private func handlePrediction(points: [LandmarkPoint]) {
        let features = LandmarkUtils.prepareForPrediction(from: points)
        
        gesturePredictor.predictWithTemporalSmoothing(from: features) { result in
            DispatchQueue.main.async {
                if let result = result {
                    self.gestureRecognized = result.label
                    self.predictionConfidence = result.confidence
                    self.isGestureRecognized = true
                } else {
                    self.gestureRecognized = "Unknown"
                    self.predictionConfidence = 0.0
                    self.isGestureRecognized = false
                }
            }
        }
    }
        
    private func updateUI(elapsed: Double, delta: Double, landmarks: [[NormalizedLandmark]],
                         gesture: String, confidence: Double, isRecognized: Bool) {
        inferenceTime = elapsed
        frameInterval = delta
        detectedHands = landmarks
        gestureRecognized = gesture
        predictionConfidence = confidence
        isGestureRecognized = isRecognized
    }
    
    func shouldSaveSample(
        newPoints: [LandmarkPoint],
        lastSaved: [LandmarkPoint]?,
        config: DefaultCostants,
        distanceThreshold: Double = 0.02,
        minVisibleLandmarks: Int = 5
    ) -> Bool {

        func handPresenceProxy(from landmarks: [LandmarkPoint]) -> Float {
            guard !landmarks.isEmpty else { return 0.0 }

            let inside = landmarks.map { (0.0...1.0).contains($0.x) && (0.0...1.0).contains($0.y) ? Float(1.0) : Float(0.0) }
            let avgInside = inside.reduce(Float(0.0), +) / Float(landmarks.count)

            return min(avgInside, 1.0)
        }

        let presence = handPresenceProxy(from: newPoints)
        
        DispatchQueue.main.async {
            self.avgPresence = presence
        }
        
        let passedPresence = presence >= config.minHandPresenceConfidence

        print("""
        [DEBUG] Presence check:
        - Numero landmarks: \(newPoints.count)
        - Proxy medio: \(String(format: "%.3f", presence))
        - Soglia: \(String(format: "%.2f", config.minHandPresenceConfidence))
        - Risultato: \(passedPresence ? "Accettato" : "Scartato")
        """)

        if !passedPresence { return false }

        let validCount = newPoints.filter { (0.0...1.0).contains($0.x) && (0.0...1.0).contains($0.y) }.count
        if validCount < minVisibleLandmarks {
            print("[DEBUG] Scartato: solo \(validCount) landmarks visibili (< \(minVisibleLandmarks))")
            return false
        }

        guard let last = lastSaved else {
            print("[DEBUG] Primo sample → salvo")
            return true
        }

        let dist = meanEuclideanDistance(newPoints, last)
        let result = dist >= distanceThreshold
        print("[DEBUG] Distanza media: \(String(format: "%.4f", dist)) → \(result ? "Accettato" : "Troppo simile")")
        return result
    }

    func meanEuclideanDistance(_ a: [LandmarkPoint], _ b: [LandmarkPoint]) -> Double {
        guard a.count == b.count, a.count > 0 else { return Double.infinity }
        var sum: Double = 0
        for i in 0..<a.count {
            let dx = Double(a[i].x - b[i].x)
            let dy = Double(a[i].y - b[i].y)
            let dz = Double(a[i].z - b[i].z)
            sum += dx*dx + dy*dy + dz*dz
        }
        return sqrt(sum) / Double(a.count)
    }
}
