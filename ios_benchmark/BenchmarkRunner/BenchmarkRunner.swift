import CoreML
import Foundation
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Argument Parsing

struct BenchmarkArgs {
    let modelPath: String
    let inputSize: Int
    let warmupIterations: Int
    let measurementIterations: Int
    let computeUnit: String

    static func parse() -> BenchmarkArgs {
        let args = CommandLine.arguments
        var modelPath = ""
        var inputSize = 640
        var warmup = 5
        var iterations = 20
        var computeUnit = "ALL"

        var i = 1
        while i < args.count {
            switch args[i] {
            case "--model":
                i += 1
                if i < args.count { modelPath = args[i] }
            case "--input-size":
                i += 1
                if i < args.count { inputSize = Int(args[i]) ?? 640 }
            case "--warmup":
                i += 1
                if i < args.count { warmup = Int(args[i]) ?? 5 }
            case "--iterations":
                i += 1
                if i < args.count { iterations = Int(args[i]) ?? 20 }
            case "--compute-unit":
                i += 1
                if i < args.count { computeUnit = args[i] }
            default:
                break
            }
            i += 1
        }

        return BenchmarkArgs(
            modelPath: modelPath,
            inputSize: inputSize,
            warmupIterations: warmup,
            measurementIterations: iterations,
            computeUnit: computeUnit
        )
    }
}

// MARK: - Device Info

struct DeviceInfo {
    let name: String
    let chip: String
    let osVersion: String

    static func current() -> DeviceInfo {
        #if os(iOS)
        let name = UIDevice.current.name
        let osVersion = UIDevice.current.systemVersion
        let chip = Self.chipName()
        #else
        let name = Host.current().localizedName ?? ProcessInfo.processInfo.hostName
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        let chip = Self.chipName()
        #endif
        return DeviceInfo(name: name, chip: chip, osVersion: osVersion)
    }

    private static func chipName() -> String {
        var size: size_t = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        let identifier = String(cString: machine)

        // Map common iPhone identifiers to chip names
        let chipMap: [String: String] = [
            "iPhone15,2": "A16 Bionic", "iPhone15,3": "A16 Bionic",
            "iPhone15,4": "A16 Bionic", "iPhone15,5": "A16 Bionic",
            "iPhone16,1": "A17 Pro", "iPhone16,2": "A17 Pro",
            "iPhone17,1": "A18 Pro", "iPhone17,2": "A18 Pro",
            "iPhone17,3": "A18", "iPhone17,4": "A18",
            "iPhone17,5": "A18", "iPhone18,1": "A19",
            "iPhone18,2": "A19 Pro", "iPhone18,3": "A19 Pro",
        ]

        return chipMap[identifier] ?? identifier
    }
}

// MARK: - Memory Tracking

func peakMemoryMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if result == KERN_SUCCESS {
        return Double(info.resident_size_max) / (1024.0 * 1024.0)
    }
    return -1.0
}

// MARK: - Input Creation

func createDummyMultiArray(shape: [Int]) -> MLMultiArray? {
    let nsShape = shape.map { NSNumber(value: $0) }
    guard let array = try? MLMultiArray(shape: nsShape, dataType: .float32) else {
        return nil
    }
    // Fill with deterministic values
    let count = shape.reduce(1, *)
    for i in 0..<count {
        array[i] = NSNumber(value: Float(i % 256) / 255.0)
    }
    return array
}

func createDummyPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width, height,
        kCVPixelFormatType_32BGRA,
        attrs as CFDictionary,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
        return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    if let baseAddress = CVPixelBufferGetBaseAddress(buffer) {
        let byteCount = CVPixelBufferGetDataSize(buffer)
        // Fill with deterministic pattern
        let ptr = baseAddress.bindMemory(to: UInt8.self, capacity: byteCount)
        for i in 0..<byteCount {
            ptr[i] = UInt8(i % 256)
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])

    return buffer
}

// MARK: - Statistics

struct LatencyStats {
    let meanMs: Double
    let medianMs: Double
    let minMs: Double
    let maxMs: Double
    let p95Ms: Double
    let stdMs: Double
    let samples: Int

    static func compute(from timings: [Double]) -> LatencyStats {
        guard !timings.isEmpty else {
            return LatencyStats(
                meanMs: 0, medianMs: 0, minMs: 0, maxMs: 0,
                p95Ms: 0, stdMs: 0, samples: 0
            )
        }

        let sorted = timings.sorted()
        let count = sorted.count
        let sum = sorted.reduce(0, +)
        let mean = sum / Double(count)

        let median: Double
        if count % 2 == 0 {
            median = (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            median = sorted[count / 2]
        }

        let p95Index = min(Int(Double(count) * 0.95), count - 1)
        let p95 = sorted[p95Index]

        let variance = sorted.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(count)
        let std = sqrt(variance)

        return LatencyStats(
            meanMs: mean, medianMs: median,
            minMs: sorted.first!, maxMs: sorted.last!,
            p95Ms: p95, stdMs: std, samples: count
        )
    }

    func toDict() -> [String: Any] {
        return [
            "mean_ms": round(meanMs * 1000) / 1000,
            "median_ms": round(medianMs * 1000) / 1000,
            "min_ms": round(minMs * 1000) / 1000,
            "max_ms": round(maxMs * 1000) / 1000,
            "p95_ms": round(p95Ms * 1000) / 1000,
            "std_ms": round(stdMs * 1000) / 1000,
            "samples": samples,
        ]
    }
}

// MARK: - Model Loading and Input Preparation

func resolveComputeUnits(_ name: String) -> MLComputeUnits {
    switch name.uppercased() {
    case "CPU_ONLY": return .cpuOnly
    case "CPU_AND_GPU": return .cpuAndGPU
    case "CPU_AND_NE": return .cpuAndNeuralEngine
    default: return .all
    }
}

func prepareInput(
    for model: MLModel, inputSize: Int
) throws -> MLFeatureProvider {
    let description = model.modelDescription

    var features: [String: MLFeatureValue] = [:]

    for (name, inputDesc) in description.inputDescriptionsByName {
        switch inputDesc.type {
        case .image:
            guard let constraint = inputDesc.imageConstraint else {
                throw BenchmarkError.inputPreparation("No image constraint for input '\(name)'")
            }
            let width = constraint.pixelsWide > 0 ? constraint.pixelsWide : inputSize
            let height = constraint.pixelsHigh > 0 ? constraint.pixelsHigh : inputSize
            guard let buffer = createDummyPixelBuffer(width: width, height: height) else {
                throw BenchmarkError.inputPreparation("Failed to create pixel buffer for '\(name)'")
            }
            features[name] = MLFeatureValue(pixelBuffer: buffer)

        case .multiArray:
            guard let constraint = inputDesc.multiArrayConstraint else {
                throw BenchmarkError.inputPreparation("No multiarray constraint for input '\(name)'")
            }
            let shape = constraint.shape.map { $0.intValue }
            guard let array = createDummyMultiArray(shape: shape) else {
                throw BenchmarkError.inputPreparation("Failed to create MLMultiArray for '\(name)'")
            }
            features[name] = MLFeatureValue(multiArray: array)

        default:
            // Provide a scalar placeholder
            features[name] = MLFeatureValue(double: 0.0)
        }
    }

    return try MLDictionaryFeatureProvider(dictionary: features)
}

// MARK: - Errors

enum BenchmarkError: Error, CustomStringConvertible {
    case modelLoad(String)
    case inputPreparation(String)
    case inference(String)

    var description: String {
        switch self {
        case .modelLoad(let msg): return "Model load error: \(msg)"
        case .inputPreparation(let msg): return "Input preparation error: \(msg)"
        case .inference(let msg): return "Inference error: \(msg)"
        }
    }
}

// MARK: - Benchmark Execution

func runBenchmark(args: BenchmarkArgs) -> [String: Any] {
    let device = DeviceInfo.current()

    // Load model
    let modelURL: URL
    if args.modelPath.hasSuffix(".mlmodelc") || args.modelPath.hasSuffix(".mlpackage") {
        modelURL = URL(fileURLWithPath: args.modelPath)
    } else {
        modelURL = URL(fileURLWithPath: args.modelPath)
    }

    let config = MLModelConfiguration()
    config.computeUnits = resolveComputeUnits(args.computeUnit)

    let model: MLModel
    do {
        model = try MLModel(contentsOf: modelURL, configuration: config)
    } catch {
        return errorResult(device: device, computeUnit: args.computeUnit,
                           message: "Failed to load model: \(error.localizedDescription)")
    }

    // Verify input preparation works before entering the benchmark loop
    do {
        _ = try prepareInput(for: model, inputSize: args.inputSize)
    } catch {
        return errorResult(device: device, computeUnit: args.computeUnit,
                           message: "Failed to prepare input: \(error)")
    }

    let totalIterations = args.warmupIterations + args.measurementIterations

    var preprocessTimings: [Double] = []
    var inferenceTimings: [Double] = []
    var postprocessTimings: [Double] = []
    var endToEndTimings: [Double] = []

    for i in 0..<totalIterations {
        let isMeasurement = i >= args.warmupIterations

        let e2eStart = CFAbsoluteTimeGetCurrent()

        // Preprocess: re-create input each iteration to measure real cost
        let preStart = CFAbsoluteTimeGetCurrent()
        let iterInput: MLFeatureProvider
        do {
            iterInput = try prepareInput(for: model, inputSize: args.inputSize)
        } catch {
            return errorResult(device: device, computeUnit: args.computeUnit,
                               message: "Input preparation failed at iteration \(i): \(error)")
        }
        let preEnd = CFAbsoluteTimeGetCurrent()

        // Inference
        let infStart = CFAbsoluteTimeGetCurrent()
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: iterInput)
        } catch {
            return errorResult(device: device, computeUnit: args.computeUnit,
                               message: "Prediction failed at iteration \(i): \(error)")
        }
        let infEnd = CFAbsoluteTimeGetCurrent()

        // Postprocess: extract output feature names and values
        let postStart = CFAbsoluteTimeGetCurrent()
        for name in prediction.featureNames {
            _ = prediction.featureValue(for: name)
        }
        let postEnd = CFAbsoluteTimeGetCurrent()

        let e2eEnd = CFAbsoluteTimeGetCurrent()

        if isMeasurement {
            preprocessTimings.append((preEnd - preStart) * 1000.0)
            inferenceTimings.append((infEnd - infStart) * 1000.0)
            postprocessTimings.append((postEnd - postStart) * 1000.0)
            endToEndTimings.append((e2eEnd - e2eStart) * 1000.0)
        }
    }

    let preprocessStats = LatencyStats.compute(from: preprocessTimings)
    let inferenceStats = LatencyStats.compute(from: inferenceTimings)
    let postprocessStats = LatencyStats.compute(from: postprocessTimings)
    let endToEndStats = LatencyStats.compute(from: endToEndTimings)

    let fps = endToEndStats.meanMs > 0 ? 1000.0 / endToEndStats.meanMs : 0.0
    let memory = peakMemoryMB()

    return [
        "success": true,
        "device_name": device.name,
        "device_chip": device.chip,
        "ios_version": device.osVersion,
        "compute_unit": args.computeUnit,
        "preprocess": preprocessStats.toDict(),
        "inference": inferenceStats.toDict(),
        "postprocess": postprocessStats.toDict(),
        "end_to_end": endToEndStats.toDict(),
        "estimated_fps": round(fps * 100) / 100,
        "peak_memory_mb": memory > 0 ? round(memory * 100) / 100 : NSNull(),
        "warmup_iterations": args.warmupIterations,
        "measurement_iterations": args.measurementIterations,
    ]
}

func errorResult(device: DeviceInfo, computeUnit: String, message: String) -> [String: Any] {
    return [
        "success": false,
        "device_name": device.name,
        "device_chip": device.chip,
        "ios_version": device.osVersion,
        "compute_unit": computeUnit,
        "error_message": message,
    ]
}

// MARK: - JSON Output

func printJSON(_ dict: [String: Any]) {
    guard let data = try? JSONSerialization.data(
        withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]
    ) else {
        let fallback = "{\"success\": false, \"error_message\": \"Failed to serialize results\"}"
        print(fallback)
        return
    }
    if let jsonString = String(data: data, encoding: .utf8) {
        print(jsonString)
    }
}

// MARK: - Entry Point

let args = BenchmarkArgs.parse()

if args.modelPath.isEmpty {
    let errorDict: [String: Any] = [
        "success": false,
        "error_message": "No model path provided. Usage: BenchmarkRunner --model <path> [--input-size N] [--warmup N] [--iterations N] [--compute-unit UNIT]",
    ]
    printJSON(errorDict)
    exit(1)
}

let result = runBenchmark(args: args)
printJSON(result)

let exitCode: Int32 = (result["success"] as? Bool) == true ? 0 : 1
exit(exitCode)
