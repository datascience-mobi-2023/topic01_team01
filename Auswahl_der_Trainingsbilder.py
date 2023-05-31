import random

def select_images(image_list, num_select):
    selected_images = random.sample(image_list, num_select)
    return selected_images

num_people = 15  # Anzahl der Personen
num_images_per_person = 11  # Anzahl der Bilder pro Person
num_select_per_person = 8  # Anzahl der ausgewählten Bilder pro Person

# Liste aller Bildnamen erstellen
all_images = []
for person in range(1, num_people + 1):
    for image in range(1, num_images_per_person + 1):
        image_name = f"subject{str(person).zfill(2)}.{str(image).zfill(2)}.gif"
        all_images.append(image_name)

# Auswahl der Bilder für jede Person
image_selection = {}
for person in range(1, num_people + 1):
    person_images = [image for image in all_images if image.startswith(f"subject{str(person).zfill(2)}.")]
    selected_images = select_images(person_images, num_select_per_person)
    image_selection[person] = selected_images

# Ausgabe der Auswahl
for person in image_selection:
    selected_images = image_selection[person]
    print(f"Person {person}:")
    print("Ausgewählte Trainingsbilder:")
    for image in selected_images:
        print(image)
    print()
