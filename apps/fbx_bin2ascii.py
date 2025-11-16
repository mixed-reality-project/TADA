# Usage:
#   python fbx_bin2ascii_sdk.py /path/in.fbx /path/out_ascii.fbx
import sys
import os
from fbx import *


def InitializeSdkObjects():
    # The first thing to do is to create the FBX SDK manager which is the
    # object allocator for almost all the classes in the SDK.
    lSdkManager = FbxManager.Create()
    if not lSdkManager:
        sys.exit(0)

    # Create an IOSettings object
    ios = FbxIOSettings.Create(lSdkManager, IOSROOT)
    lSdkManager.SetIOSettings(ios)

    # Create the entity that will hold the scene.
    lScene = FbxScene.Create(lSdkManager, "")

    return (lSdkManager, lScene)

def SaveScene(pSdkManager, pScene, pFilename, pFileFormat=-1, pEmbedMedia=False):
    lExporter = FbxExporter.Create(pSdkManager, "")
    if pFileFormat < 0 or pFileFormat >= pSdkManager.GetIOPluginRegistry().GetWriterFormatCount():
        pFileFormat = pSdkManager.GetIOPluginRegistry().GetNativeWriterFormat()
        if not pEmbedMedia:
            lFormatCount = pSdkManager.GetIOPluginRegistry().GetWriterFormatCount()
            for lFormatIndex in range(lFormatCount):
                if pSdkManager.GetIOPluginRegistry().WriterIsFBX(lFormatIndex):
                    lDesc = pSdkManager.GetIOPluginRegistry().GetWriterFormatDescription(lFormatIndex)
                    if "ascii" in lDesc:
                        pFileFormat = lFormatIndex
                        break

    if not pSdkManager.GetIOSettings():
        ios = FbxIOSettings.Create(pSdkManager, IOSROOT)
        pSdkManager.SetIOSettings(ios)

    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_MATERIAL, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_TEXTURE, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, pEmbedMedia)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_SHAPE, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_GOBO, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_ANIMATION, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_GLOBAL_SETTINGS, True)

    result = lExporter.Initialize(pFilename, pFileFormat, pSdkManager.GetIOSettings())
    if result == True:
        result = lExporter.Export(pScene)

    lExporter.Destroy()
    return result


def LoadScene(pSdkManager, pScene, pFileName):
    lImporter = FbxImporter.Create(pSdkManager, "")
    result = lImporter.Initialize(pFileName, -1, pSdkManager.GetIOSettings())
    if not result:
        return False

    if lImporter.IsFBX():
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_MATERIAL, True)
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_TEXTURE, True)
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, True)
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_SHAPE, True)
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_GOBO, True)
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_ANIMATION, True)
        pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_GLOBAL_SETTINGS, True)

    result = lImporter.Import(pScene)
    lImporter.Destroy()
    return result


def find_ascii_writer_format_id(manager):
    """
    Return the writer format ID whose description contains 'ASCII'.
    Falls back to the default writer if not found (but that should exist on standard installs).
    """
    registry = manager.GetIOPluginRegistry()
    count = registry.GetWriterFormatCount()
    default_id = registry.GetNativeWriterFormat()
    ascii_id = None
    for i in range(count):
        if registry.WriterIsFBX(i):
            desc = registry.GetWriterFormatDescription(i) or ""
            if "ascii" in desc.lower():
                ascii_id = i
                break
    return ascii_id if ascii_id is not None else default_id

def convert_to_ascii(src_path, dst_path):
    # Create manager/scene
    manager, scene = InitializeSdkObjects()
    try:
        # Import
        status = LoadScene(manager, scene, src_path)
        if not status:
            raise RuntimeError(f"Failed to load FBX: {src_path}")

        # Prepare exporter
        exporter = FbxExporter.Create(manager, "")
        writer_id = find_ascii_writer_format_id(manager)

        ios = manager.GetIOSettings()
        if ios is None:
            ios = FbxIOSettings.Create(manager, IOSROOT)
            manager.SetIOSettings(ios)

        # Initialize exporter for ASCII
        if not exporter.Initialize(dst_path, writer_id, ios):
            raise RuntimeError(f"Exporter init failed: {exporter.GetStatus().GetErrorString()}")

        # (Optional) ensure ASCII flag if the writer supports it (some SDK versions obey this)
        # ios.SetBoolProp(fbx.EXP_FBX_ASCII, True)  # Not all Python bindings expose this; writer_id is more reliable.

        # Export
        if not exporter.Export(scene):
            raise RuntimeError(f"Export failed: {exporter.GetStatus().GetErrorString()}")

        exporter.Destroy()
        print(f"Converted:\n  IN  = {src_path}\n  OUT = {dst_path}")
    finally:
        manager.Destroy()

def main():
    if len(sys.argv) < 3:
        print("Usage: python fbx_bin2ascii_sdk.py <in.fbx> <out_ascii.fbx>")
        sys.exit(2)
    src = os.path.abspath(sys.argv[1])
    dst = os.path.abspath(sys.argv[2])
    convert_to_ascii(src, dst)

if __name__ == "__main__":
    main()
