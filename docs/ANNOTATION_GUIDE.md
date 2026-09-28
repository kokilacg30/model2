# Street AIQ — Road-Trash Annotation Guide & Protocol

**Project Target:** Single-Class Road-Trash Detection & Instance Segmentation  
**Class Name:** `trash`  
**Class ID:** `0`  
**Annotation Type:** Polygon Instance Segmentation (CVAT / Roboflow preferred)

---

## 1. Project Goal & Core Rule

The objective of **Street AIQ** is to detect **ONLY actual litter/trash on or immediately adjacent to road surfaces**.

* **Single Class Standard:** Every trash instance, regardless of material (plastic, paper, metal, glass, organic waste pile), MUST be annotated under the single class name **`trash`** (Class ID `0`).
* **Human Verification Required:** Do NOT use AI-generated auto-ground-truth. Every polygon must be human-verified.

---

## 2. What Counts as Trash vs. What Does NOT

### A. What COUNTS as Trash (`trash` / Class ID 0)
* **Plastic:** Bottles, plastic bags, wrappers, packaging films, cups, straws, fast-food containers, disposable cutlery, plastic jugs.
* **Paper & Cardboard:** Cartons, crushed paper, flyers, cardboard boxes discarded on the road, newspapers.
* **Metals:** Aluminum cans, tin foil, discarded metal scrap, bottle caps.
* **Glass:** Glass bottles, broken glass containers on the road.
* **Textiles & Household Waste:** Old clothes, shoes, rags, sponges, discarded household items.
* **Garbage Piles:** Accumulated heaps of mixed urban waste on road asphalt, curbs, or sidewalks immediately touching the road.

### B. What DOES NOT Count as Trash (DO NOT ANNOTATE)
* **Vehicles:** Cars, trucks, buses, motorcycles, bicycles, auto-rickshaws.
* **People & Animals:** Pedestrians, stray animals.
* **Natural Elements:** Live plants, standing trees, grass, soil, mud, rocks, stones, loose gravel.
* **Leaves & Twigs:** Fallen leaves/branches UNLESS they are explicitly mixed into an artificial garbage heap.
* **Infrastructure & Road Features:** Potholes, asphalt patches, manhole covers, speed bumps, road lane markings, paint strips, curbs, sidewalk tiles, drainage grates, utility poles, walls, buildings.
* **Shadows & Lighting:** Dark shadows cast by trees, buildings, or vehicles.

---

## 3. Detailed Handling Rules for 11 Scenarios

### Scenario 1: What Counts as Trash
Annotate any man-made waste object or discarded material present on the road carriageway, shoulder, curb, or adjacent gutter.

### Scenario 2: What Does NOT Count as Trash
Never annotate normal urban environment elements (potholes, leaves, rocks, shadows, road markings, vehicles, people). If an item is ambiguous (e.g. a dark patch on asphalt), verify whether it has 3D volume, plastic sheen, or clear waste geometry before labeling.

### Scenario 3: Partially Visible Trash
If a trash item is partially occluded by a car tire, curb, or shadow, **annotate only the visible portion** using a tight polygon around the visible pixels. Do NOT estimate or extrapolate hidden boundaries under tires or vehicles.

### Scenario 4: Overlapping Trash Items
* **Distinct / Separable Items:** If individual bottles or wrappers overlap but their boundaries are visually distinct, draw separate tight polygon annotations for each item.
* **Dense / Indistinguishable Heap:** If items are fused together in a dense pile where individual borders cannot be determined, treat the entire pile as a single `trash` polygon (see Scenario 5).

### Scenario 5: Garbage Piles
When multiple trash items form an aggregate pile:
* Draw a single unified polygon around the perimeter of the entire garbage pile.
* Ensure the polygon outline tightly hugs the outer edges of the waste, excluding clean asphalt or adjacent vegetation.

### Scenario 6: Very Small Trash
* Annotate small trash items (e.g., small candy wrappers, cigarette boxes, bottle caps) if they are clearly recognizable at 100% zoom (>= 10x10 pixels).
* If an item is smaller than 10x10 pixels or appears as an unidentifiable 2-pixel spec, **do NOT annotate it** to avoid noise.

### Scenario 7: Blurry Trash
* If an image exhibits motion blur or focus blur:
  * Annotate the item ONLY if its identity as trash is reasonably certain.
  * Draw the polygon around the visible blurred body.
  * If blur makes it impossible to distinguish between a leaf, a shadow, or a wrapper, **leave it unannotated**.

### Scenario 8: Uncertain / Ambiguous Objects
If you cannot determine whether an object is trash or a natural element (e.g., a white stone vs. a crumpled paper cup):
* Apply the **"Strict Verification Rule"**: When in doubt, **do NOT annotate**. False positive annotations in ground truth degrade model performance much more severely than a missing ambiguous instance.

### Scenario 9: Trash Near Vehicles
* If a bottle or bag is lying near a car tire or under a parked vehicle, annotate the trash tight to its visible borders.
* Do NOT include vehicle tires, bumpers, or shadow pixels inside the `trash` polygon.

### Scenario 10: Trash Near Plants / Foliage
* Trash often accumulates near bushes or grass along road edges.
* Draw the polygon strictly around the artificial trash pixels. Exclude grass blades, soil, and leaves surrounding the plastic/paper item.

### Scenario 11: Trash at the Road Boundary / Curb
* Trash located on the curb line, gutter, or sidewalk edge directly adjacent to the driving lane MUST be annotated.
* Ensure the polygon boundary cuts precisely along the edge of the trash object, excluding curb concrete or sidewalk brickwork.

---

## 4. Polygon Quality Standards

1. **Tightness:** Polygons must tightly bound the trash object. No loose rectangular boxes for irregular shapes when using segmentation mode.
2. **Point Density:** Use enough vertices to capture curves (typically 8 to 20 points per object). Avoid excessive points (>50) for simple rectangular wrappers.
3. **No Overlap with Non-Trash:** Ensure background asphalt, grass, or shadow pixels are excluded from the polygon interior.

---

## 5. Summary Checklist for Annotators

Before submitting an annotated frame, verify:
- [ ] Are all visible trash items labeled as Class `0` (`trash`)?
- [ ] Are clean roads left completely empty (0 labels)?
- [ ] Are potholes, shadows, rocks, leaves, and road markings left unannotated?
- [ ] Are garbage piles tightly outlined?
- [ ] Are polygon coordinates normalized between 0.0 and 1.0 upon export?
