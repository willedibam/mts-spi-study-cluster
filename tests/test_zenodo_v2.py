"""Geometry, historical compatibility and offline export integrity for v2."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from scripts.explore_zenodo_v2 import corrected_tags, dataset_origin, distance_neighbors, marker_radii, medoids_alternate, summarize_partition


def test_distance_neighbors_exclude_self_for_exact_duplicates():
    x = np.array([[0., 0.], [0., 0.], [2., 0.], [4., 0.]])
    nn = distance_neighbors(squareform(pdist(x)), 2)
    assert all(i not in row for i, row in enumerate(nn))
    assert nn[0, 0] == 1 and nn[1, 0] == 0


def test_full_rank_unwhitened_pca_is_full_distance_geometry():
    x = np.random.default_rng(5).normal(size=(12, 40))
    scores = PCA(svd_solver="full", whiten=False).fit_transform(x)[:, :11]
    np.testing.assert_allclose(pdist(x), pdist(scores), atol=1e-12)
    assert not np.allclose(pdist(x), pdist(scores[:, :2]))


def test_alternate_medoids_reaches_real_members_and_nearest_assignment():
    x = np.array([[0., 0.], [0., 1.], [0., 2.], [10., 0.], [10., 1.], [10., 2.]])
    labels, centers, iterations = medoids_alternate(x, 2)
    assert len(np.unique(centers)) == 2
    assert iterations < 300
    d = squareform(pdist(x))
    np.testing.assert_array_equal(labels, d[centers].argmin(axis=0))
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        assert centers[group] in members
        costs = d[np.ix_(members, members)].sum(axis=1)
        assert costs[members.tolist().index(centers[group])] == costs.min()


def test_partition_diagnostics_exclude_noise_and_use_full_space():
    x = np.array([[0., 0.], [0., 1.], [0., 2.], [10., 0.], [10., 1.], [50., 50.]])
    result = summarize_partition(np.array([0, 0, 0, 1, 1, -1]), squareform(pdist(x)))
    assert result["clusters"] == 2
    assert result["coverage"] == 5 / 6
    assert result["groups"]["0"]["medoid"] == 1
    assert result["full_silhouette_assigned"] > .8
    json.dumps(result, allow_nan=False)


def test_noise_only_partition_is_reported_without_nan():
    result = summarize_partition(np.full(4, -1), np.zeros((4, 4)))
    assert result["clusters"] == 0
    assert result["coverage"] == 0
    assert result["full_silhouette_assigned"] is None
    json.dumps(result, allow_nan=False)


def test_template_preserves_offline_inspection_controls():
    template = (Path(__file__).parents[1] / "scripts/zenodo_gallery_v2.html").read_text()
    assert template.count("__CORPUS_DATA__") == 1
    assert "fetch(" not in template
    for control in ["download", "export-svg", "export-png", "permalink", "choose", "clear", "search", "cross-map"]:
        assert f'id="{control}"' in template


def test_plot_and_exports_have_square_aspect():
    template = (Path(__file__).parents[1] / "scripts/zenodo_gallery_v2.html").read_text()
    assert 'viewBox="0 0 680 680"' in template
    assert 'aspect-ratio:1' in template
    assert '[0,0,680,510]' not in template
    assert "setAttribute('width','1360');clone.setAttribute('height','1360')" in template
    assert 'c.width=2040;c.height=2040' in template
    assert "d:'M60 35 V615 H640 V35 Z'" in template


def test_marker_area_increases_with_m_and_uses_fixed_corpus_bounds():
    radii = marker_radii([5, 17, 29], 5, 29)
    np.testing.assert_allclose(radii ** 2, [4, 10, 16])
    assert marker_radii([17], 5, 29)[0] == radii[1]
    assert marker_radii([5], 5, 5)[0] == 2
    template = (Path(__file__).parents[1] / "scripts/zenodo_gallery_v2.html").read_text()
    assert 'r:r.marker_radius' in template
    assert 'r:focused===row' not in template


def test_origin_uses_archive_tags_without_guessing_unlabelled_sources():
    assert dataset_origin(["real", "finance"]) == "real"
    assert dataset_origin(["synthetic", "wave"]) == "synthetic"
    assert dataset_origin(["real", "synthetic"]) == "mixed"
    assert dataset_origin(["fmri", "HCP"]) == "unlabelled"
    assert dataset_origin([]) == "unlabelled"


def test_label_corrections_append_only_to_matching_datasets():
    rules = [{"pattern": "hcp_tfMRI_*", "tags": ["real"]},
             {"pattern": "hcp_rsfMRI_*", "tags": ["real"]}]
    original = ["fmri", "neuroscience"]
    for name in ["hcp_tfMRI_S0_R-115-124", "hcp_rsfMRI_S8_R-53-62"]:
        result = corrected_tags(name, original, rules)
        assert result == original + ["real"]
        assert corrected_tags(name, result, rules) == result
        assert dataset_origin(result) == "real"
    assert original == ["fmri", "neuroscience"]
    assert corrected_tags("other_fmri", original, rules) == original


def test_offline_kde_contours_and_unassigned_exclusion():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the standalone KDE renderer")
    script = Path(__file__).parents[1] / "scripts/zenodo_kde.js"
    check = """
const {assignedGroups,densityContours}=require(process.argv[1]);
const points=[[0,0],[1,0],[0,1],[1,1],[.4,.6],[.6,.3],[999,999],[-999,-999]];
const groups=assignedGroups(points,points.map((_,i)=>i),[0,0,0,0,0,0,-1,-2]);
if(JSON.stringify(Object.keys(groups))!=='["0"]'||groups[0].length!==6)throw Error('Noise pooled into KDE');
const contours=densityContours(groups[0]);
if(contours.length!==2||contours[0].mass!==.8||contours[1].mass!==.5)throw Error('Missing contours');
for(const c of contours)if(!c.path.startsWith('M')||!c.path.endsWith('Z')||/NaN|Infinity/.test(c.path))throw Error('Invalid path');
if(densityContours([[0,0],[0,0],[0,0]]).length)throw Error('Singular KDE');
if(densityContours([[0,0],[1,1],[2,2]]).length)throw Error('Collinear KDE');
if(densityContours([[0,0],[1,1]]).length)throw Error('Insufficient points');
function area(path){return path.split('M').filter(Boolean).reduce((sum,s)=>{const a=s.slice(0,-1).split('L').map(p=>p.split(',').map(Number));return sum+Math.abs(a.reduce((v,p,i)=>{const q=a[(i+1)%a.length];return v+p[0]*q[1]-q[0]*p[1]},0)/2)},0)}
if(!(area(contours[0].path)>area(contours[1].path)))throw Error('Mass contours not nested');
console.log('KDE paths, mass ordering, degeneracies, negative-label exclusion pass');
"""
    result = subprocess.run([node, "-e", check, str(script)], capture_output=True, text=True, check=True)
    assert "exclusion pass" in result.stdout


def test_kde_controls_and_unassigned_click_selects_its_group():
    template = (Path(__file__).parents[1] / "scripts/zenodo_gallery_v2.html").read_text()
    assert template.count("__KDE_SCRIPT__") == 1
    assert '<option value="kde">KDE</option>' in template
    assert "if(p)$('cluster').value=String(p.labels[row]);" in template
    assert "if(p&&p.labels[row]<0){detail();draw();return}" not in template
    assert "stroke:'#ffffff'" in template
