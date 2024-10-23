document.addEventListener('DOMContentLoaded', function() {
    // 지도 초기화
    var map = L.map('map').setView([37.5665, 126.9780], 12); // Default center to Seoul

    // 타일 레이어 추가
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
    }).addTo(map);

    // 서버에서 전달된 쉼터 데이터
    var shelters = window.sheltersData || [];

    // 함수: 마커와 정보 패널을 업데이트
    function updateMapAndInfo(shelters) {
        // 기존 레이어 제거
        map.eachLayer(function(layer) {
            if (!!layer.toGeoJSON) {
                map.removeLayer(layer);
            }
        });

        // 정보 패널 초기화
        var infoContainer = document.getElementById('info-container');
        infoContainer.innerHTML = '';

        // 새 마커와 정보 패널 추가
        shelters.forEach(function(shelter) {
            if (shelter.latitude && shelter.longitude) {
                var marker = L.marker([shelter.latitude, shelter.longitude]).addTo(map);
                marker.bindPopup(`<b>${shelter.facility_name}</b><br>${shelter.address}<br>Tel: ${shelter.tel}`);

                // 정보 패널에 추가
                var shelterInfo = document.createElement('div');
                shelterInfo.className = 'shelter-info';
                shelterInfo.innerHTML = `<strong>${shelter.facility_name}</strong><br>
                                          주소: ${shelter.address}<br>
                                          전화번호: ${shelter.tel}<br>
                                          위도: ${shelter.latitude}<br>
                                          경도: ${shelter.longitude}`;
                infoContainer.appendChild(shelterInfo);
            }
        });
    }

    // 초기화: 모든 쉼터 표시
    updateMapAndInfo(shelters);

    // 이벤트 핸들러: 지역 선택
    document.getElementById('region').addEventListener('change', function(event) {
        var selectedRegion = event.target.value;
        var filteredShelters = shelters.filter(function(shelter) {
            return shelter.district_name === selectedRegion || selectedRegion === "";
        });
        updateMapAndInfo(filteredShelters);
    });

    // 이벤트 핸들러: 대피소명 검색
    document.getElementById('shelter').addEventListener('input', function(event) {
        var searchTerm = event.target.value.toLowerCase();
        var filteredShelters = shelters.filter(function(shelter) {
            return shelter.facility_name.toLowerCase().includes(searchTerm);
        });
        updateMapAndInfo(filteredShelters);
    });
});
