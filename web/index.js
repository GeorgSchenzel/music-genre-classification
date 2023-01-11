const input = document.getElementById("song-file-input");
const result = document.getElementById("result");
const chartCanvas = document.getElementById('result-chart');

const chart = new Chart(chartCanvas, {
    type: 'bar',
    data: {
        labels: [],
        datasets: []
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
    chart.data.labels =  Object.keys(data)
    chart.data.datasets[0] = {
        data: Object.values(data),
        borderWidth: 0
    };
    chart.update();
}

const onchange = () => {
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
            updateChart(success["all_preds"]);
        }
    ).catch(
        error => console.log(error)
    );
};

input.addEventListener('change', () => onchange(), false);

