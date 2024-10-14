import os
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import dotenv

dotenv.load_dotenv()


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_epub(epub_path, output_dir):
    book = epub.read_epub(epub_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chapter_number = 1
    for item in book.get_items():
        # Check if the item is of type 'EpubHtml' which represents HTML documents
        if isinstance(item, epub.EpubHtml):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            content = clean_text(soup.get_text())

            if content.strip() and len(content) > 100:
                filename = f"{chapter_number:04d}_{content[:20]}.txt"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                print(f"Saved chapter: {filename}")
                chapter_number += 1


if __name__ == "__main__":
    epub_path = os.getenv("NOVEL_FILE")
    output_dir = os.getenv("NOVEL_FILE_DIR")
    parse_epub(epub_path, output_dir)
    print("EPUB parsing completed!")
