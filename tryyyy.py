import

{useState, useRef, useCallback}
from

"react";
import

{Button}
from

"@/components/ui/button";
import

{Card, CardContent}
from

"@/components/ui/card";
import

{Camera, Upload, X, Loader2}
from

"lucide-react";
import

{cn}
from

"@/lib/utils";

interface
CameraCaptureProps
{
    onImageCapture: (imageData: string) = > void;
isAnalyzing: boolean;
}

const
CameraCapture = ({onImageCapture, isAnalyzing}: CameraCaptureProps) = > {
    const[isCameraActive, setIsCameraActive] = useState(false);
const[stream, setStream] = useState < MediaStream | null > (null);
const
videoRef = useRef < HTMLVideoElement > (null);
const
canvasRef = useRef < HTMLCanvasElement > (null);
const
fileInputRef = useRef < HTMLInputElement > (null);

const
startCamera = useCallback(async () = > {
try {
const mediaStream = await navigator.mediaDevices.getUserMedia({
video: {
           facingMode: "environment", // Use
back
camera
on
mobile
width: {ideal: 1920},
height: {ideal: 1080}
}
});

if (videoRef.current) {
videoRef.current.srcObject = mediaStream;
setStream(mediaStream);
setIsCameraActive(true);
}
} catch (error) {
console.error("Error accessing camera:", error);
}
}, []);

const
stopCamera = useCallback(() = > {
if (stream) {
stream.getTracks().forEach(track = > track.stop());
setStream(null);
}
setIsCameraActive(false);
}, [stream]);

const
capturePhoto = useCallback(() = > {
if (videoRef.current & & canvasRef.current) {
const canvas = canvasRef.current;
const video = videoRef.current;

canvas.width = video.videoWidth;
canvas.height = video.videoHeight;

const ctx = canvas.getContext("2d");
if (ctx) {
ctx.drawImage(video, 0, 0);
const imageData = canvas.toDataURL("image/jpeg", 0.8);
onImageCapture(imageData);
stopCamera();
}
}
}, [onImageCapture, stopCamera]);

const
handleFileUpload = useCallback((event: React.ChangeEvent < HTMLInputElement >) = > {
const
file = event.target.files?.[0];
if (file & & file.type.startsWith("image/")) {
const reader = new FileReader();
reader.onload = (e) = > {
const result = e.target?.result as string;
onImageCapture(result);
};
reader.readAsDataURL(file);
}
}, [onImageCapture]);

return (
    < div className="space-y-4" >
    {!isCameraActive ? (
    < div className="grid grid-cols-1 md:grid-cols-2 gap-4" >
    < Card className="bg-gradient-leaf border-leaf/20" >
    < CardContent className="p-6 text-center" >
    < Camera className="w-12 h-12 mx-auto mb-4 text-primary-foreground" / >
    < h3 className="text-lg font-semibold text-primary-foreground mb-2" >
    Take Photo
    < / h3 >
    < p className="text-primary-foreground/80 mb-4 text-sm" >
    Capture a live image of your plant for analysis
    < / p >
    < Button
    onClick={startCamera}
    disabled={isAnalyzing}
    className="w-full bg-primary-foreground text-primary hover:bg-primary-foreground/90"
    >
    {isAnalyzing ? (
    < Loader2 className="w-4 h-4 mr-2 animate-spin" / >
): (
    < Camera className="w-4 h-4 mr-2" / >
)}
Start
Camera
< / Button >
    < / CardContent >
        < / Card >

            < Card
className = "bg-gradient-earth border-earth/20" >
            < CardContent
className = "p-6 text-center" >
            < Upload
className = "w-12 h-12 mx-auto mb-4 text-primary-foreground" / >
            < h3
className = "text-lg font-semibold text-primary-foreground mb-2" >
            Upload
Image
< / h3 >
    < p
className = "text-primary-foreground/80 mb-4 text-sm" >
            Select
an
existing
photo
from your device
< / p >
    < Button
onClick = {() = > fileInputRef.current?.click()}
disabled = {isAnalyzing}
className = "w-full bg-primary-foreground text-primary hover:bg-primary-foreground/90"
            >
            {isAnalyzing ? (
    < Loader2 className="w-4 h-4 mr-2 animate-spin" / >
): (
    < Upload className="w-4 h-4 mr-2" / >
)}
Choose
File
< / Button >
    < input
ref = {fileInputRef}
type = "file"
accept = "image/*"
onChange = {handleFileUpload}
className = "hidden"
            / >
            < / CardContent >
                < / Card >
                    < / div >
): (
    < Card className="relative overflow-hidden bg-black" >
    < CardContent className="p-0 relative" >
    < video
    ref={videoRef}
    autoPlay
    playsInline
    className="w-full aspect-video object-cover"
    / >

    {/ * Scanning overlay * /}
    < div className="absolute inset-0 pointer-events-none" >
    < div className="absolute inset-4 border-2 border-healthy rounded-lg" >
    < div className="absolute top-0 left-0 w-6 h-6 border-t-4 border-l-4 border-healthy rounded-tl-lg" > < / div >
    < div className="absolute top-0 right-0 w-6 h-6 border-t-4 border-r-4 border-healthy rounded-tr-lg" > < / div >
    < div className="absolute bottom-0 left-0 w-6 h-6 border-b-4 border-l-4 border-healthy rounded-bl-lg" > < / div >
    < div className="absolute bottom-0 right-0 w-6 h-6 border-b-4 border-r-4 border-healthy rounded-br-lg" > < / div >
    < / div >
    < div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" >
    < div className="w-2 h-32 bg-healthy/50 animate-pulse" > < / div >
    < / div >
    < / div >

    < div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-4" >
    < Button
    onClick={capturePhoto}
    size="lg"
    className={cn(
    "bg-healthy hover:bg-healthy/90 text-white",
    "shadow-glow animate-pulse-glow"
)}
>
< Camera
className = "w-5 h-5 mr-2" / >
            Capture
            < / Button >
                < Button
onClick = {stopCamera}
size = "lg"
variant = "destructive"
          >
          < X
className = "w-5 h-5 mr-2" / >
            Cancel
            < / Button >
                < / div >
                    < / CardContent >
                        < / Card >
)}

< canvas
ref = {canvasRef}
className = "hidden" / >
            < / div >
);
};

export
default
CameraCapture;