export type QuarterTurn = 0 | 90 | 180 | 270;

export function rotateQuarterTurn(current: QuarterTurn, delta: -90 | 90): QuarterTurn {
  return ((current + delta + 360) % 360) as QuarterTurn;
}

function rotatedFileName(file: File): string {
  const stem = file.name.replace(/\.[^.]+$/, "") || "road-sign";
  return `${stem}.oriented.jpg`;
}

export async function bakeImageRotation(file: File, rotation: QuarterTurn): Promise<File> {
  if (rotation === 0) return file;

  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  try {
    const swapAxes = rotation === 90 || rotation === 270;
    const canvas = document.createElement("canvas");
    canvas.width = swapAxes ? bitmap.height : bitmap.width;
    canvas.height = swapAxes ? bitmap.width : bitmap.height;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("Image rotation is unavailable in this browser.");

    context.translate(canvas.width / 2, canvas.height / 2);
    context.rotate((rotation * Math.PI) / 180);
    context.drawImage(bitmap, -bitmap.width / 2, -bitmap.height / 2);
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (value) => value ? resolve(value) : reject(new Error("Unable to encode rotated image.")),
        "image/jpeg",
        0.95,
      );
    });
    return new File([blob], rotatedFileName(file), {
      type: "image/jpeg",
      lastModified: file.lastModified,
    });
  } finally {
    bitmap.close();
  }
}
