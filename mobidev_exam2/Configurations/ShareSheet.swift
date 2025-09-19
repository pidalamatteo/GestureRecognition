import SwiftUI
import UIKit


extension View {
    func presentShareSheet(items: [Any], isPresented: Binding<Bool>) -> some View {
        self.background(
            ShareSheetHelper(items: items, isPresented: isPresented)
                .frame(width: 0, height: 0)
        )
    }
}

struct ShareSheetHelper: UIViewControllerRepresentable {
    let items: [Any]
    @Binding var isPresented: Bool

    func makeUIViewController(context: Context) -> UIViewController {
        UIViewController()
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        guard isPresented && !items.isEmpty else { return }

        let activityVC = UIActivityViewController(activityItems: items, applicationActivities: nil)
        activityVC.popoverPresentationController?.sourceView = uiViewController.view

        uiViewController.present(activityVC, animated: true) {
            isPresented = false
        }
    }
}
