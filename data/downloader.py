import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

device_names = [
    "D01_Samsung_GalaxyS3Mini",
    "D02_Apple_iPhone4s",
    "D05_Apple_iPhone5c",
    "D06_Apple_iPhone6",
    "D09_Apple_iPhone4",
    "D14_Apple_iPhone5c",
    "D15_Apple_iPhone6",
    "D18_Apple_iPhone5c",
    "D26_Samsung_GalaxyS3Mini",
    "D29_Apple_iPhone5",
    "D34_Apple_iPhone5",
]

# URL of the main page
main_url = "https://lesc.dinfo.unifi.it/VISION/dataset/"

# Send a GET request to the main page
response = requests.get(main_url)

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

def download_images(d):
    # Find the anchor tag with the matching href
    anchor = soup.find("a", href=str(d) + "/")

    if anchor:
        # Construct the URL for the device page
        device_url = main_url + anchor["href"]

        # Send a GET request to the device page
        device_response = requests.get(device_url)
        device_soup = BeautifulSoup(device_response.content, "html.parser")

        # Find the "images/" link
        images_link = device_soup.find("a", string="images/")

        if images_link:
            # Construct the URL for the images page
            images_url = device_url + images_link["href"]

            # Send a GET request to the images page
            images_response = requests.get(images_url)
            images_soup = BeautifulSoup(images_response.content, "html.parser")

            # Find the "flat/" link
            flat_link = images_soup.find("a", string="flat/")

            if flat_link:
                # Construct the URL for the flat page
                flat_url = images_url + flat_link["href"]

                # Send a GET request to the flat page
                flat_response = requests.get(flat_url)
                flat_soup = BeautifulSoup(flat_response.content, "html.parser")

                # Find all image links on the flat page
                image_links = flat_soup.find_all("a", href=lambda href: href and href.endswith('.jpg'))

                if image_links:
                    # Create the folder for the device
                    folder_name = "./data/devices/" + str(d) + "/flat"
                    os.makedirs(folder_name, exist_ok=True)

                    # Download each image
                    for link in image_links:
                        image_url = flat_url + link["href"]
                        image_response = requests.get(image_url)
                        print(f"Downloaded {image_url}")

                        # Save the image to the device folder
                        image_name = link["href"]
                        image_path = os.path.join(folder_name, image_name)
                        with open(image_path, "wb") as file:
                            file.write(image_response.content)
                    print(f"Downloaded flat images for device {d}")
                else:
                    print(f"No flat image links found for device {d}")
            else:
                print(f"No 'flat/' link found for device {d}")

            # Find the "nat/" link
            nat_link = images_soup.find("a", string="nat/")

            if nat_link:
                # Construct the URL for the nat page
                nat_url = images_url + nat_link["href"]

                # Send a GET request to the nat page
                nat_response = requests.get(nat_url)
                nat_soup = BeautifulSoup(nat_response.content, "html.parser")

                # Find all image links on the nat page
                image_links = nat_soup.find_all("a", href=lambda href: href and href.endswith('.jpg'))

                if image_links:
                     # Create the folder for the device
                    folder_name = "./data/devices/" + str(d) + "/nat"
                    os.makedirs(folder_name, exist_ok=True)

                    # Download each image
                    for link in image_links:
                        image_url = nat_url + link["href"]
                        image_response = requests.get(image_url)
                        print(f"Downloaded {image_url}")

                        # Save the image to the device folder
                        image_name = link["href"]
                        image_path = os.path.join(folder_name, image_name)
                        with open(image_path, "wb") as file:
                            file.write(image_response.content)
                else:
                    print(f"No nat image links found for device {d}")

                print(f"Downloaded natural images for device {d}")
            else:
                print(f"No 'nat/' link found for device {d}")
        else:
            print(f"No 'images/' link found for device {d}")
    else:
        print(f"No anchor found for device {d}")

with ThreadPoolExecutor() as executor:
    # Submit the download_images function for each D value
    futures = [executor.submit(download_images, d) for d in device_names]

    # Wait for all the futures to complete
    for future in futures:
        future.result()
