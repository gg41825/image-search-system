import re
import json
import time
import requests
from bs4 import BeautifulSoup

# Mapping between category display name and URL slug
categories_map = {
    "Outdoor Jacken": "shirts-hemden-longsleeves",
    "Outdoor Josen": "outdoor-hosen",
    "Shirts, Hemden & Longsleeves": "shirts-hemden-longsleeves",
    "Kopfbedeckungen": "kopfbedeckungen",
    "Pullover & Hoodies": "pullover-hoodies",
    "Funktionsunterwäsche": "funktionsunterwaesche",  # replace special chars
    "Bademode": "bademode",
    "Outdoor Westen": "outdoor-westen",
    "Brillen": "brillen",
    "Kleider & Röcke": "kleider-roecke",
    "Handschuhe": "handschuhe",
    "Overalls": "overalls",
    "Accessoires": "accessoires",
    "Textilpflege": "textilpflege"
}

base_url = "https://www.bergfreunde.de"
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; MyCrawler/1.0; +https://example.com/bot)"
}


def clean_price(raw_price: str) -> float | None:
    """Convert raw price string like '€ 54,97' into float 54.97."""
    if not raw_price:
        return None
    cleaned = re.sub(r"[^\d,]", "", raw_price)  # keep only digits and comma
    return float(cleaned.replace(",", ".")) if cleaned else None


def scrape_category(category_name: str, slug: str, _id_start: int = 1):
    """
    Scrape products from one category page.

    Args:
        category_name (str): Display name of the category.
        slug (str): URL slug for the category.
        _id_start (int): Starting ID for products.

    Returns:
        tuple[list[dict], int]: A list of product dicts and the next available ID.
    """
    url = f"{base_url}/{slug}/"
    print(f"🔗 Requesting: {url}")
    products_data = []
    current_id = _id_start

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find product list container
        product_list = soup.find("ul", id="product-list")
        if not product_list:
            print("⚠️ No product-list found")
            return products_data, current_id

        # Find all product-infobox blocks inside product-list
        products = product_list.find_all("div", class_="product-infobox")

        for product in products:
            try:
                # Extract brand
                brand = product.find("div", class_="manufacturer-title")
                brand_name = brand.get_text(strip=True) if brand else ""

                # Extract product title
                title_div = product.find("div", class_="product-title")
                title_parts = [t.get_text(strip=True) for t in title_div.find_all("span")] if title_div else []
                product_title = " ".join(title_parts)

                # Extract product image (inside <a class="product-link">)
                link_tag = product.find_parent("a", class_="product-link")
                img = link_tag.find("img", class_="product-image") if link_tag else None
                img_url = img["src"] if img else ""

                # Extract price (sibling of product-infobox)
                price_div = product.find_next_sibling("div", class_="product-price")
                raw_price = price_div.find("span", {"data-codecept": "currentPrice"}).get_text(strip=True) if price_div else ""
                price = clean_price(raw_price)

                # Build product dictionary
                product_doc = {
                    "_id": str(current_id),
                    "name": f"{brand_name} - {product_title}".strip(),
                    "category": category_name,
                    "price": price,
                    "image_url": img_url,
                }
                products_data.append(product_doc)
                current_id += 1

            except Exception as e:
                print(f"⚠️ Error parsing product: {e}")

    except requests.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")

    return products_data, current_id


def main():
    """Main crawler function."""
    all_products = []
    _id = 1

    for category_name, slug in categories_map.items():
        products, _id = scrape_category(category_name, slug, _id)
        all_products.extend(products)
        time.sleep(3)  # Sleep between requests

    # Save results into JSON
    with open("app/data/products.json", "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_products)} products to products.json")


if __name__ == "__main__":
    main()