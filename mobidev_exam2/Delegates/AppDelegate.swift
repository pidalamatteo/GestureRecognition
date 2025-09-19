//
//  AppDelegate.swift
//  mobidev_exam2
//
//  Created by Matteo on 09/09/25.
//
import SwiftUI

// AppDelegate "vecchio stile" se serve per librerie esterne
class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil
    ) -> Bool {
        print("AppDelegate did finish launching")
        LandmarkUtils.loadSelectedFeatureIndices()
        //DEBUG
          LandmarkUtils.verifyNormalizationConsistency()
          LandmarkUtils.verifyFeatureIndices()
          LandmarkUtils.verifyFeatureConsistency()
        return true
    }
}

@main
struct MyApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView() // La root SwiftUI View
        }
    }
}


