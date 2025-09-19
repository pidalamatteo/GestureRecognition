//
//  ContentView.swift
//  mobidev_exam2
//
//  Created by Matteo on 07/05/25.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        ZStack{
            CameraView()
                .ignoresSafeArea()
        }
    }
}
#Preview {
    ContentView()
}
