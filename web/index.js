const input = document.getElementById("song-file-input");
const result = document.getElementById("result");
const uploadButton = document.getElementById("upload-button");
const fileName = document.getElementById("file-name");
const chartCanvas = document.getElementById('result-chart');

const chart = new Chart(chartCanvas, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{}]
    },
    options: {
        scales: {
            x: {
                grid: {
                    display: false
                },
                border: {
                    display: false
                }
            },
            y: {
                beginAtZero: true,
                display: false,
                min: 0,
                max: 1
            }
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                enabled: false
            }
        }
    }
});

const updateChart = (data) => {
    chart.data.labels = Object.keys(data)
    chart.data.datasets[0] = {
        data: Object.values(data),
        backgroundColor: "#1095c1"
    };
    chart.update();
}

const resetChart = () => {
    chart.data.datasets[0].data = Array(chart.data.labels.length).fill(0);
    chart.update();
}

const onchange = () => {
    resetChart();
    fileName.innerText = input.files[0].name;
    result.innerText = "";
    result.setAttribute("aria-busy", "true");
    uploadButton.setAttribute("disabled", "");

    const formData = new FormData();
    formData.append('file', input.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    }).then(
        response => response.json()
    ).then(
        success => {
            result.textContent = success["class_name"];
            result.setAttribute("aria-busy", "false");;
                uploadButton.removeAttribute("disabled");

            updateChart(success["all_preds"]);
        }
    ).catch(
        error => console.log(error)
    );
};

input.addEventListener('change', () => onchange(), false);

const genres = document.getElementById("supported-genres");
fetch("/genres").then(
    response => response.json()
).then(
    success => {
        genres.textContent = success.join(", ");
    });
