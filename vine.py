import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
from rasterio.features import rasterize
from skimage import measure, filters, morphology
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from rasterio.warp import transform_geom

ORTHO_PATH = '/home/praho/Documents/Job/BlajADER/HartiParcele/107Media-orthophoto.tif'
ROWS_PATH = '/home/praho/Documents/Job/BlajADER/Randuri/107randuri.geojson'

src = rasterio.open(ORTHO_PATH)
r, g, b = src.read(1), src.read(2), src.read(3)
h, w = src.height, src.width
transform = src.transform

rows = gpd.read_file(ROWS_PATH)
if 'row_id' not in rows.columns:
    rows['row_id'] = range(1, len(rows) + 1)

print(f"✅ Loaded orthophoto: {w}x{h} pixels")
print(f"✅ Loaded {len(rows)} rows")
print(f"📍 CRS: {src.crs}")


def calculate_vegetation_indices(r, g, b):
    r_norm = r.astype(float) / 255.0
    g_norm = g.astype(float) / 255.0
    b_norm = b.astype(float) / 255.0

    epsilon = 1e-7
    indices = {}
    indices['exg'] = 2 * g_norm - r_norm - b_norm
    excess_red = 1.4 * r_norm - g_norm
    indices['exgr'] = indices['exg'] - excess_red
    indices['ndi'] = (g_norm - r_norm) / (g_norm + r_norm + epsilon)
    indices['vari'] = (g_norm - r_norm) / (g_norm + r_norm - b_norm + epsilon)
    indices['grvi'] = (g_norm - r_norm) / (g_norm + r_norm + epsilon)
    indices['rgbvi'] = (g_norm ** 2 - b_norm * r_norm) / (g_norm ** 2 + b_norm * r_norm + epsilon)
    return indices


def detect_bare_soil_kmeans(r, g, b, n_clusters=4):
    pixels = np.stack([r.flatten(), g.flatten(), b.flatten()], axis=1)

    valid_mask = (pixels[:, 0] > 10) & (pixels[:, 0] < 245) & \
                 (pixels[:, 1] > 10) & (pixels[:, 1] < 245) & \
                 (pixels[:, 2] > 10) & (pixels[:, 2] < 245)

    valid_pixels = pixels[valid_mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(valid_pixels)

    labels = np.full(len(pixels), -1)
    labels[valid_mask] = kmeans.predict(valid_pixels)
    labels = labels.reshape(h, w)

    centroids = kmeans.cluster_centers_

    soil_scores = []
    for i, centroid in enumerate(centroids):
        r_val, g_val, b_val = centroid
        soil_score = (r_val - g_val) + (r_val - b_val) + (g_val - b_val) * 0.5
        brightness = (r_val + g_val + b_val) / 3
        if 80 < brightness < 180:
            soil_score += 20
        soil_scores.append(soil_score)

    soil_cluster = np.argmax(soil_scores)
    bare_soil_mask = (labels == soil_cluster)

    return bare_soil_mask, labels, centroids


def pixel_to_geographic(row, col, transform):
    lon, lat = transform * (col, row)
    return lon, lat


# Funcție pentru calcularea distanței în metri între două puncte GPS
def calculate_distance_meters(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # Raza Pământului în metri

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


print("🧮 Calculating vegetation indices...")
indices = calculate_vegetation_indices(r, g, b)

print("🎯 Applying K-means clustering...")
bare_soil_mask, labels, centroids = detect_bare_soil_kmeans(r, g, b)

print(f"📊 Cluster centroids (RGB):")
for i, centroid in enumerate(centroids):
    print(f"  Cluster {i}: R={centroid[0]:.1f}, G={centroid[1]:.1f}, B={centroid[2]:.1f}")


def create_gap_mask(indices, bare_soil_mask, approach='combined_improved'):
    if approach == 'exg_adaptive':
        threshold = np.percentile(indices['exg'], 25)
        return indices['exg'] < threshold

    elif approach == 'exgr':
        threshold = np.percentile(indices['exgr'], 20)
        return indices['exgr'] < threshold

    elif approach == 'vari':
        threshold = np.percentile(indices['vari'], 30)
        return indices['vari'] < threshold

    elif approach == 'kmeans':
        return bare_soil_mask

    elif approach == 'combined_improved':
        exg_mask = indices['exg'] < np.percentile(indices['exg'], 25)
        vari_mask = indices['vari'] < np.percentile(indices['vari'], 25)
        exgr_mask = indices['exgr'] < np.percentile(indices['exgr'], 20)
        combined_mask = bare_soil_mask | exg_mask | vari_mask | exgr_mask
        combined_mask = morphology.binary_opening(combined_mask, morphology.disk(1))
        combined_mask = morphology.binary_closing(combined_mask, morphology.disk(2))
        combined_mask = morphology.remove_small_objects(combined_mask, min_size=2)
        return combined_mask


approaches = ['exg_adaptive', 'exgr', 'vari', 'kmeans', 'combined_improved']
approach_results = {}

for approach in approaches:
    gap_mask = create_gap_mask(indices, bare_soil_mask, approach)
    gap_percentage = np.sum(gap_mask) / gap_mask.size * 100
    approach_results[approach] = {
        'mask': gap_mask,
        'percentage': gap_percentage
    }
    print(f"📈 {approach}: {gap_percentage:.1f}% potential gaps")

best_approach = 'combined_improved'
final_gap_mask = approach_results[best_approach]['mask']

print(f"✅ Using {best_approach} approach: {approach_results[best_approach]['percentage']:.1f}% gaps")

print(f"\n🔄 Processing {len(rows)} rows...")
gaps = []
row_summary = []
MIN_AREA = 15
MAX_AREA = 3000

print(f"🎯 Căutăm doar gap-uri ≥ {MIN_AREA} pixeli (dimensiune considerabilă)")

for idx, row in rows.iterrows():
    try:
        mask = rasterize(
            [(row.geometry, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        if np.sum(mask) == 0:
            continue

        row_gaps_mask = np.logical_and(final_gap_mask, mask == 1)
        labels = measure.label(row_gaps_mask, connectivity=2)
        props = measure.regionprops(labels)

        row_gaps = []
        gap_coordinates = []
        total_components = len(props)
        valid_components = 0

        for gap_id, prop in enumerate(props, 1):
            if prop.area < MIN_AREA:
                continue
            if prop.area > MAX_AREA:
                continue

            valid_components += 1
            centroid_row, centroid_col = prop.centroid
            gap_lon, gap_lat = pixel_to_geographic(centroid_row, centroid_col, src.transform)

            minr, minc, maxr, maxc = prop.bbox
            lon_min, lat_max = src.transform * (minc, minr)
            lon_max, lat_min = src.transform * (maxc, maxr)

            # Calculăm dimensiunile în metri
            width_meters = calculate_distance_meters(lat_min, lon_min, lat_min, lon_max)
            height_meters = calculate_distance_meters(lat_min, lon_min, lat_max, lon_min)
            area_sqm = width_meters * height_meters

            poly = box(lon_min, lat_min, lon_max, lat_max)
            gap_info = {
                'row_id': row.row_id,
                'gap_id': valid_components,
                'geometry': poly,
                'centroid_point': Point(gap_lon, gap_lat),
                'centroid_lon': gap_lon,
                'centroid_lat': gap_lat,
                'area_pixels': prop.area,
                'area_sqm': area_sqm,
                'width_meters': width_meters,
                'height_meters': height_meters,
                'bbox_lon_min': lon_min,
                'bbox_lat_min': lat_min,
                'bbox_lon_max': lon_max,
                'bbox_lat_max': lat_max
            }
            gaps.append(gap_info)
            row_gaps.append(gap_info)
            gap_coordinates.append({
                'gap_id': valid_components,
                'lon': gap_lon,
                'lat': gap_lat,
                'pixels': prop.area,
                'area_sqm': area_sqm,
                'width_m': width_meters,
                'height_m': height_meters
            })

        if total_components > 0:
            print(
                f"🔍 Randul {row.row_id}: {total_components} componente găsite, {valid_components} gap-uri semnificative (≥{MIN_AREA} pixeli)")

        if len(row_gaps) > 0:
            print(f"\n📍 RANDUL {row.row_id}:")
            print(f"   🔢 Numărul de gap-uri semnificative: {len(row_gaps)}")
            print(f"   📍 Coordonate GPS ale gap-urilor:")

            for gap_coord in gap_coordinates:
                print(
                    f"      Gap {gap_coord['gap_id']}: {gap_coord['lat']:.6f}°N, {gap_coord['lon']:.6f}°E (dimensiune: {gap_coord['pixels']} pixeli, {gap_coord['area_sqm']:.1f} m²)")

            row_summary.append({
                'row_id': row.row_id,
                'gap_count': len(row_gaps),
                'gap_coordinates': gap_coordinates
            })

    except Exception as e:
        print(f"❌ Error processing row {row.row_id}: {e}")
        continue

print(f"\n🎉 Rezultat final:")
print(f"   Prag minim pentru gap-uri: {MIN_AREA} pixeli")
print(f"   Total gap-uri semnificative detectate: {len(gaps)}")
print(f"   Rânduri cu gap-uri semnificative: {len(row_summary)}/{len(rows)}")

if gaps:
    gdf = gpd.GeoDataFrame(gaps, crs=src.crs)
    OUT_GEOJSON = 'vineyard_gaps_detailed.geojson'
    gdf.to_file(OUT_GEOJSON, driver='GeoJSON')

    gap_points = []
    for gap in gaps:
        gap_points.append({
            'row_id': gap['row_id'],
            'gap_id': gap['gap_id'],
            'geometry': gap['centroid_point'],
            'lon': gap['centroid_lon'],
            'lat': gap['centroid_lat'],
            'pixels': gap['area_pixels'],
            'area_sqm': gap['area_sqm']
        })

    gdf_points = gpd.GeoDataFrame(gap_points, crs=src.crs)
    OUT_POINTS = 'vineyard_gap_centers.geojson'
    gdf_points.to_file(OUT_POINTS, driver='GeoJSON')

    print(f"\n🎉 REZULTATE FINALE:")
    print(f"📊 Total gap-uri detectate: {len(gaps)}")
    print(f"📈 Rânduri cu gap-uri: {len(row_summary)}/{len(rows)}")
    print(f"💾 Salvat în: {OUT_GEOJSON}")
    print(f"💾 Centre gap-uri salvate în: {OUT_POINTS}")

    # SUMARIZARE DETALIATĂ ÎMBUNĂTĂȚITĂ
    summary_filename = 'sumarizare_randuri_detaliata.txt'
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("╔" + "═" * 80 + "╗\n")
        f.write("║" + " " * 25 + "SUMARIZARE DETALIATĂ PE RÂNDURI" + " " * 24 + "║\n")
        f.write("║" + " " * 80 + "║\n")
        f.write("║" + f" Total rânduri analizate: {len(rows)}" + " " * (55 - len(str(len(rows)))) + "║\n")
        f.write("║" + f" Rânduri cu gap-uri: {len(row_summary)}" + " " * (59 - len(str(len(row_summary)))) + "║\n")
        f.write("║" + f" Total gap-uri detectate: {len(gaps)}" + " " * (56 - len(str(len(gaps)))) + "║\n")
        f.write("║" + f" Prag minim: {MIN_AREA} pixeli" + " " * (59 - len(str(MIN_AREA))) + "║\n")
        f.write("║" + f" CRS: {src.crs}" + " " * (76 - len(str(src.crs))) + "║\n")
        f.write("╚" + "═" * 80 + "╝\n\n")

        # Statistici generale
        if row_summary:
            total_gaps = sum(row_info['gap_count'] for row_info in row_summary)
            total_area_sqm = sum(gap['area_sqm'] for gap in gaps)
            avg_gaps_per_row = total_gaps / len(row_summary)
            max_gaps_row = max(row_summary, key=lambda x: x['gap_count'])

            f.write("📊 STATISTICI GENERALE:\n")
            f.write("─" * 50 + "\n")
            f.write(f"• Gap-uri per rând (medie): {avg_gaps_per_row:.1f}\n")
            f.write(
                f"• Rândul cu cele mai multe gap-uri: {max_gaps_row['row_id']} ({max_gaps_row['gap_count']} gap-uri)\n")
            f.write(f"• Suprafața totală a gap-urilor: {total_area_sqm:.1f} m²\n")
            f.write(f"• Suprafața medie per gap: {total_area_sqm / len(gaps):.1f} m²\n\n")

        f.write("📍 DETALII COMPLETE PE RÂNDURI:\n")
        f.write("═" * 80 + "\n\n")

        for row_info in sorted(row_summary, key=lambda x: x['row_id']):
            f.write(f"🌿 RÂNDUL {row_info['row_id']}\n")
            f.write("─" * 40 + "\n")
            f.write(f"📊 Numărul total de gap-uri: {row_info['gap_count']}\n\n")

            total_area_row = sum(gap['area_sqm'] for gap in row_info['gap_coordinates'])
            f.write(f"📐 Suprafața totală gap-uri pe rând: {total_area_row:.1f} m²\n\n")

            f.write("📍 COORDONATE GPS DETALIATE:\n")
            f.write("┌" + "─" * 78 + "┐\n")
            f.write("│ Gap │      Latitudine      │      Longitudine     │  Suprafața │ Dimensiuni │\n")
            f.write("│ ID  │         (°N)         │         (°E)         │    (m²)    │    (m)     │\n")
            f.write("├" + "─" * 78 + "┤\n")

            for gap_coord in sorted(row_info['gap_coordinates'], key=lambda x: x['gap_id']):
                f.write(
                    f"│ {gap_coord['gap_id']:2d}  │ {gap_coord['lat']:15.8f}    │ {gap_coord['lon']:15.8f}    │ {gap_coord['area_sqm']:8.1f}   │ {gap_coord['width_m']:.1f}x{gap_coord['height_m']:.1f}  │\n")

            f.write("└" + "─" * 78 + "┘\n\n")

            # Coordonate în formatul Google Maps
            f.write("🗺️  COORDONATE PENTRU GOOGLE MAPS:\n")
            for gap_coord in sorted(row_info['gap_coordinates'], key=lambda x: x['gap_id']):
                f.write(f"   Gap {gap_coord['gap_id']:2d}: {gap_coord['lat']:.6f}, {gap_coord['lon']:.6f}\n")

            f.write("\n" + "═" * 80 + "\n\n")

        # Rezumat final
        f.write("📋 REZUMAT EXECUTIV:\n")
        f.write("─" * 50 + "\n")
        f.write(f"Analiza a identificat {len(gaps)} gap-uri semnificative distribuite pe {len(row_summary)} rânduri.\n")
        if row_summary:
            f.write(f"Suprafața medie a unui gap este de {total_area_sqm / len(gaps):.1f} m².\n")
            f.write(
                f"Rândul cu cele mai multe probleme este rândul {max_gaps_row['row_id']} cu {max_gaps_row['gap_count']} gap-uri.\n")
        f.write(
            f"\nToate gap-urile au o suprafață minimă de {MIN_AREA} pixeli pentru a fi considerate semnificative.\n")

    print(f"💾 Sumarizare detaliată salvată în: {summary_filename}")

    # RAPORT CSV PENTRU IMPORT ÎN EXCEL/GIS
    csv_filename = 'gap_coordinates_detailed.csv'
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write("row_id,gap_id,latitude,longitude,area_pixels,area_sqm,width_m,height_m,google_maps_link\n")

        for row_info in sorted(row_summary, key=lambda x: x['row_id']):
            for gap_coord in sorted(row_info['gap_coordinates'], key=lambda x: x['gap_id']):
                google_maps_link = f"https://maps.google.com/?q={gap_coord['lat']:.6f},{gap_coord['lon']:.6f}"
                f.write(
                    f"{row_info['row_id']},{gap_coord['gap_id']},{gap_coord['lat']:.8f},{gap_coord['lon']:.8f},{gap_coord['pixels']},{gap_coord['area_sqm']:.1f},{gap_coord['width_m']:.1f},{gap_coord['height_m']:.1f},{google_maps_link}\n")

    print(f"💾 Date CSV salvate în: {csv_filename}")

    # RAPORT JSON PENTRU APLICAȚII
    import json

    json_filename = 'gap_data_complete.json'
    json_data = {
        'metadata': {
            'total_rows': len(rows),
            'rows_with_gaps': len(row_summary),
            'total_gaps': len(gaps),
            'min_area_pixels': MIN_AREA,
            'crs': str(src.crs),
            'processing_date': str(np.datetime64('today'))
        },
        'statistics': {
            'total_area_sqm': sum(gap['area_sqm'] for gap in gaps),
            'average_gaps_per_row': sum(row_info['gap_count'] for row_info in row_summary) / len(
                row_summary) if row_summary else 0,
            'average_gap_area_sqm': sum(gap['area_sqm'] for gap in gaps) / len(gaps) if gaps else 0
        },
        'rows': []
    }

    for row_info in sorted(row_summary, key=lambda x: x['row_id']):
        row_data = {
            'row_id': row_info['row_id'],
            'gap_count': row_info['gap_count'],
            'total_area_sqm': sum(gap['area_sqm'] for gap in row_info['gap_coordinates']),
            'gaps': []
        }

        for gap_coord in sorted(row_info['gap_coordinates'], key=lambda x: x['gap_id']):
            gap_data = {
                'gap_id': gap_coord['gap_id'],
                'coordinates': {
                    'latitude': gap_coord['lat'],
                    'longitude': gap_coord['lon']
                },
                'dimensions': {
                    'area_pixels': gap_coord['pixels'],
                    'area_sqm': gap_coord['area_sqm'],
                    'width_m': gap_coord['width_m'],
                    'height_m': gap_coord['height_m']
                },
                'google_maps_url': f"https://maps.google.com/?q={gap_coord['lat']:.6f},{gap_coord['lon']:.6f}"
            }
            row_data['gaps'].append(gap_data)

        json_data['rows'].append(row_data)

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Date JSON salvate în: {json_filename}")

    gaps_per_row = gdf.groupby('row_id').size()
    print(f"\n📊 Statistici generale:")
    print(f"   Gap-uri per rând (medie): {gaps_per_row.mean():.1f}")
    print(f"   Rândul cu cele mai multe gap-uri: {gaps_per_row.idxmax()} ({gaps_per_row.max()} gap-uri)")
    print(f"   Suprafața totală gap-uri: {sum(gap['area_sqm'] for gap in gaps):.1f} m²")

else:
    print("❌ Nu s-au detectat gap-uri!")
    print("🔍 Informații pentru debugging:")
    print(f"   Total pixeli potențiali gap: {np.sum(final_gap_mask)}")
    print(f"   Pragul minim: {MIN_AREA} pixeli")
    print(f"   Încercați să reduceți MIN_AREA sau să ajustați pragurile vegetației")

print(f"\n💾 Salvare vizualizare debug...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

rgb_img = np.stack([r, g, b], axis=2)
axes[0, 0].imshow(rgb_img)
axes[0, 0].set_title('Original RGB')
axes[0, 0].axis('off')

axes[0, 1].imshow(final_gap_mask, cmap='Reds')
axes[0, 1].set_title(f'Gap Detection ({best_approach})')
axes[0, 1].axis('off')

axes[1, 0].imshow(labels, cmap='tab10')
axes[1, 0].set_title('K-means Clusters')
axes[1, 0].axis('off')

axes[1, 1].imshow(indices['exg'], cmap='RdYlGn')
axes[1, 1].set_title('Excess Green Index')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('gap_detection_debug.png', dpi=150, bbox_inches='tight')
print("💾 Imagine debug salvată ca 'gap_detection_debug.png'")

src.close()
print("\n✅ Procesare completă!")
print(f"📁 Fișiere generate:")
print(f"   - {OUT_GEOJSON} (gap-uri ca poligoane)")
print(f"   - {OUT_POINTS} (centre gap-uri ca puncte)")
print(f"   - {summary_filename} (raport text detaliat)")
print(f"   - {csv_filename} (date CSV pentru Excel/GIS)")
print(f"   - {json_filename} (date JSON pentru aplicații)")
print(f"   - gap_detection_debug.png (vizualizare debug)")