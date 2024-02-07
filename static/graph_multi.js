var chart;
var firstTime=true;
function requestData() {
    $.ajaxSetup({
    async: true
    });

    $.ajax({
        url: '/live-data_multi',
        success: function(point) {
            console.log("786 POINT",point)
            if(firstTime==true){
			  point.forEach((j, i) => {
					chart.addSeries(j);
			  });
			  firstTime=false;
			  }
			  else{
			    point.forEach((j, i) => {
					chart.series[i].setData(j.data, true);
			  });
			  }
        },
        cache: false
    });
}

$(document).ready(function () {
    chart = new Highcharts.Chart({
        chart: {
            type: 'packedbubble',
            renderTo: 'data-container_multi',
            events: {
                load: requestData
            },
            backgroundColor: '#002147',
            
        },
        credits: {
            enabled: false
        },
        title: {
            text: null
        },        
        tooltip: {
            useHTML: true,
            pointFormat: '<b>{point.name}:</b> {point.value}%'
        },
        plotOptions: {
            packedbubble: {
                minSize: '20%',
                maxSize: '150%',
                zMin: 0,
                zMax: 1000,
                layoutAlgorithm: {
                    gravitationalConstant: 0.05,
                    splitSeries: true,
                    seriesInteraction: false,
                    dragBetweenSeries: true,
                    parentNodeLimit: true
                },

                dataLabels: {
                    enabled: true,
                    format: '{point.name}',
                    filter: {
                        property: 'y',
                        operator: '>',
                        value: 0
                    },
                    style: {
                        color: 'black',
                        textOutline: 'none',
                        fontWeight: 'normal'
                    }
                }
            }
        },
        xAxis:[ {
                crosshair: {
                		snap: false,
                    label: {
                        enabled: true,
                        padding: 8
                    }
                }
            }],
        series: [],

    });
}

);