import Metal
import Satin

open class NearestDepthProcessor: DepthPassProcessor {
    public var winnerBuffer: MTLBuffer? {
        get { pixelBuffer }
        set { pixelBuffer = newValue }
    }
}
