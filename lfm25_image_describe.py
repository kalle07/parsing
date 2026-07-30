# ---------------- Libraries ----------------
"""
Image Description Generator using Multimodal Model
Processes images ONE AT A TIME to minimize memory usage.
Resizes only the current image before passing to LLM, then cleans up.
Compatible with Windows.
"""

import os
import gc
import io
import sys
import time
import re
from pathlib import Path
from base64 import b64encode
from typing import Optional
from PIL import Image
from llama_cpp import Llama
import llama_cpp
from llama_cpp.llama_chat_format import LFM25VLChatHandler # Gemma4ChatHandler # Qwen35ChatHandler # Qwen35ChatHandler #MTMDChatHandler
from llama_cpp import llama_print_system_info


print("=== Python ===")
print("Version:", sys.version)
print("Executable:", sys.executable)
print(llama_print_system_info().decode("utf-8"))
print("LLAMA Version:", llama_cpp.__version__)


# ---------------- Configuration ----------------
def get_config() -> dict:
    """Returns configuration for model paths and settings."""
    # === CHANGE THIS PATH TO ANY LOCATION ON YOUR HARD DRIVE ===
    model_path = Path(r"f:\path\model.gguf")
    image_path = Path(r"c:\images")
    
    # Auto-discover the mmproj file (first file containing "mmproj")
    mmproj_candidates = [f for f in model_path.parent.glob("*mmproj*.gguf") if f.is_file()]
    if not mmproj_candidates:
        raise FileNotFoundError(f"No mmproj file found in {model_path.parent}")
    
    return {
        "base_path": model_path.parent,
        "gguf_model_path": model_path,  # The model file you specified
        "clip_model_path": mmproj_candidates[0],  # Auto-discovered mmproj
        "image_input_path": image_path, # image path 
        "image_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.webp'},
        "min_side_size": 256,
        "recursive": True,
    }


# ---------------- Image Processing Functions ----------------


def find_image_files(input_path: Path, extensions: set[str], recursive: bool = True) -> list[Path]:
    """Find all image files in the specified directory."""
    images = []
    
    if not input_path.exists():
        print(f"⚠️  Input path does not exist: {input_path}")
        return images
    
    # Search pattern based on recursive setting
    search_pattern = "**/*" if recursive else "*"
    
    for file_path in input_path.glob(search_pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            images.append(file_path)
    
    return sorted(images)

    
def resize_image_to_max_side(image_path: Path, max_side: int = 256) -> Image.Image | None:
    """
    Resize image so the LARGEST side is exactly `max_side` pixels.
    
    Memory Note: Returns a new PIL Image that should be explicitly 
    released after use to prevent memory buildup in long loops.
    
    Args:
        image_path: Path to the source image file  
        max_side: Target dimension for the larger side (default: 1024)
        
    Returns:
        Resized PIL Image or None if processing fails
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            
            print(f"   ℹ️  Original size: {original_width}x{original_height}")
            
            # Target the LARGER dimension to max_side
            max_dim = max(original_width, original_height)
            scale_factor = max_side / max_dim
            
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            print(f"   ↕️  Scaling by: {scale_factor:.2f}x")
            print(f"   📐 New size: {new_width}x{new_height}")
            
            resized_img = img.resize(
                (new_width, new_height), 
                Image.Resampling.LANCZOS
            )
            
            return resized_img
            
    except Exception as e:
        print(f"⚠️  Failed to resize {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None



def image_to_base64(image: Image.Image, format_type: str = 'JPEG') -> str | None:
    """Convert a PIL Image to base64-encoded data URI string."""
    try:
        if not hasattr(image, 'filename'):
            image.filename = "unknown"
        
        mime_types = {
            'JPEG': 'image/jpeg',
            'PNG': 'image/png',
            'WEBP': 'image/webp',
            'BMP': 'image/bmp',
            'TIFF': 'image/tiff',
        }
        
        output_format = format_type.upper()
        if output_format in ['JPG', 'JPEG']:
            mime_type = 'image/jpeg'
            # JPEG doesn't support transparency - convert RGBA to RGB
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
        else:
            mime_type = mime_types.get(output_format, 'image/jpeg')
        

        byte_buffer = io.BytesIO()
        quality = 97 if output_format in ['JPG', 'JPEG'] else None
        
        save_kwargs = {'format': output_format}
        if quality is not None:
            save_kwargs['quality'] = quality
            
        image.save(byte_buffer, **save_kwargs)
        base64_data = b64encode(byte_buffer.getvalue()).decode('ascii')
        
        return f"data:{mime_type};base64,{base64_data}"
        
    except Exception as e:
        print(f"⚠️  Failed to encode {getattr(image, 'filename', 'unknown')} as base64: {e}")
        return None


# ---------------- Image Processing Functions ----------------
# In generate_description function, add explicit cleanup at function start
def generate_description(llm: Llama, image_path: Path, seed: int = 5678) -> str | None:
    """Generate a description for a single image using the multimodal model."""
    # Force memory cleanup at start of function
    gc.collect()
    
    try:
        print(f"   📐 Step 1/3: Resizing...")
        resized_image: Optional[Image.Image] = resize_image_to_max_side(
            image_path, max_side=1024
        )
        
        if resized_image is None:
            return "Failed to process image"
        
        pixel_count = resized_image.width * resized_image.height
        print(f"   🔐 Step 2/3: Converting to base64... ({pixel_count:,} pixels)")
        
        base64_uri: Optional[str] = image_to_base64(resized_image, format_type='JPEG')
        
        if not base64_uri:
            return "Failed to encode image"
        
        print(f"   🤖 Step 3/3: Generating description...")
        
        messages = [
            {
                "role": "system",
                "content": (
                    ##"You are an assistant who helps users describe images. You respond in English in a human readable format.\n"
                    "The image type can be any visual format used to present data, concepts, structures, or objects — such as charts, diagrams, drawing, table, photograph, or other illustrative graphics.\n"
                    ## "The image can be a diagram, flowchart, sketch, drawing, organisational chart or graphic, rarely text or table or a photograph.\n"
                    ## "If the image type is a table or flowchart, activate OCR tool and mention the text in addition to the description.\n"
                    "Structure of the description: image type, title or figure name, the image captioning, any further details.\n"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_uri}
                    }
                ],
            },
        ]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=800,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            min_p=0.05,
            seed=seed,
        )
        choices = response.get("choices", [])
        if not choices:
            return "No description generated"
            
        content = choices[0].get("message", {}).get("content")
        
        # === FIXED: Explicit cleanup BEFORE returning ===
        del resized_image
        del base64_uri
        del messages
        del response
        gc.collect()
        
        return content.strip() if content else "No description generated"
    except Exception as e:
        print(f"❌ Error processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_response(image_path: Path, response: str) -> bool:
    """Save the model's response to a .txt file with same basename as image."""
    # CLEAN THE DESCRIPTION BEFORE SAVING
    cleaned_response = clean_description(response)
    
    output_path = image_path.with_suffix(".txt")
    
    try:
        # Write description with UTF-8 encoding for international characters
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_response)
        print(f"✅ Saved to: {output_path.name}")
        return True
    except PermissionError as e:
        print(f"❌ Cannot write to {output_path}: {e}")
        return False
    except OSError as e:
        print(f"❌ OS error writing to {output_path}: {e}")
        return False



def clean_description(description: str) -> str:
    """
    Clean generated description by extracting text between [think] markers.
    
    Args:
        description: Raw description text from the model
        
    Returns:
        Cleaned description text between markers
    """
    if not description:
        return description
    # remove all think pattern
    # description = re.sub(r'<think>.*?</think>', '', description, flags=re.DOTALL | re.IGNORECASE)
    # description = re.sub(r'<|channel>.*?<channel|>', '', description, flags=re.DOTALL | re.IGNORECASE)
    
    
    description = re.sub(r'\*\*', '', description, flags=re.DOTALL | re.IGNORECASE)
    
    description = re.sub(r'#', '', description, flags=re.DOTALL | re.IGNORECASE)
    

    # Remove any leftover stray tags
    #description = re.sub(r'</?think>', '', description, flags=re.IGNORECASE)
    
    return description.strip()


# ---------------- Main Execution ----------------
def main():
    """Main function to process all images and generate descriptions."""
    config = get_config()
    start_time = time.time()    
    
    if not config["gguf_model_path"].exists():
        raise FileNotFoundError(
            f"Model file not found: {config['gguf_model_path'].resolve()}"
        )
    if not config["clip_model_path"].exists():
        raise FileNotFoundError(
            f"MMProj file not found: {config['clip_model_path'].resolve()}"
        )
    
    image_files = find_image_files(
        input_path=config["image_input_path"],
        extensions=config["image_extensions"],
        recursive=config["recursive"]
    )
    print(f"number of images: {len(image_files)}")
    
    if not image_files:
        print(f"No images found in {config['image_input_path']}")
        return
    
    # Base seed for LLM generation
    base_seed = 5678
    max_retries = 3
    min_char_count = 100 # 200 should be minimum
    max_char_count = 3500 # ~750 token 
    
    llm = Llama(
        model_path=str(config["gguf_model_path"]),
        n_ctx=2048,
        ctx_checkpoints=0,
        n_threads=os.cpu_count() or 8,
        n_gpu_layers=-1,
        main_gpu=0,
        use_mmap=True,
        use_mlock=False,
        low_vram=False,
        swa_full=True,
        seed=base_seed,
        verbose=False,
        chat_handler=LFM25VLChatHandler(
            clip_model_path=str(config["clip_model_path"]),
            image_min_tokens=1024,
            image_max_tokens=2048,
            keep_past_thinking=False,
        ),
    )
     
    print("✅ Model loaded successfully!\n")
    print("=" * 50)
    
    processed = 0
    failed = 0
    short_desc_count = 0
    long_desc_count = 0
    valid_desc_count = 0
    failed_image_names: list[str] = []  # Track failed image filenames
    
    for image_path in image_files:
        try:
            print(f"\n🖼️ Processing: {image_path.name}")
            
            best_description: str | None = None
            best_description_length = 0
            current_seed = base_seed
            description_found = False  # Flag to track if valid description was found
            
            for attempt in range(max_retries):
                print(f"   🔁 Attempt {attempt + 1}/{max_retries} (seed={current_seed})")
                
                description = generate_description(llm, image_path, seed=current_seed)
                
                if description:
                    desc_length = len(description)
                    print(f"   📊 Description length: {desc_length} characters")
                    
                    # === NEW: Stop immediately if within valid range ===
                    if min_char_count <= desc_length <= max_char_count:
                        best_description = description
                        best_description_length = desc_length
                        description_found = True
                        print(f"   ✅ Valid description found ({desc_length} chars)")
                        break  # Exit retry loop immediately
                    
                    # Track longest description if no valid one yet
                    if desc_length > best_description_length:
                        best_description = description
                        best_description_length = desc_length
                        print(f"   ⚠️  Attempt {attempt + 1}: {desc_length} chars (outside range)")
                
                # Increment seed for next iteration
                current_seed += 1
            
            # === Print summary of all attempts for this image ===
            print(f"   📋 Final description length: {best_description_length if best_description else 0} chars")
        
        except Exception as e:
            print(f"❌ Critical error processing {image_path.name}: {e}")
            failed += 1
            failed_image_names.append(image_path.name)
            continue
        
        # === Save response if description was found ===
        # === FIXED: Counters incremented once per image, outside retry loop ===
        if description_found and best_description:
            cleaned_response = clean_description(best_description)
            desc_len = len(cleaned_response)
            
            # === MOVED OUTSIDE: Count this image ONCE based on final result ===
            if desc_len < min_char_count:
                short_desc_count += 1
                print(f"   ⚠️  Short description saved anyway: {desc_len} chars")
            elif desc_len > max_char_count:
                long_desc_count += 1
                print(f"   ⚠️  Long description saved anyway: {desc_len} chars")
            else:
                valid_desc_count += 1
                print(f"   ✅ Description within range: {desc_len} chars")
            
            if not save_response(image_path, cleaned_response):
                failed += 1
                failed_image_names.append(image_path.name)
                continue
            
            processed += 1
        else:
            failed += 1
            failed_image_names.append(image_path.name)
            print(f"   ❌ No valid description found for {image_path.name}")
        
        # Force garbage collection after each image
        gc.collect()
        print("-" * 50)
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {processed} succeeded, {failed} failed")
    print(f"   ✅ Valid descriptions ({min_char_count}-{max_char_count} chars): {valid_desc_count}")
    print(f"   ⚠️  Short descriptions (<{min_char_count} chars): {short_desc_count}")
    print(f"   ⚠️  Long descriptions (>{max_char_count} chars): {long_desc_count}")
    
    # === NEW: Print summary of failed image filenames ===
    if failed_image_names:
        print(f"\n❌ Failed Images ({len(failed_image_names)}):")
        for name in failed_image_names:
            print(f"   - {name}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"   ⚡ Total duration:         {total_duration:.2f}s")
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"   💾 Current memory usage:  {mem_mb:.1f} MB")
    except ImportError:
        pass
    
    gc.collect()


    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except FileNotFoundError as e:
        print(f"\n❌ Configuration error: {e}")


