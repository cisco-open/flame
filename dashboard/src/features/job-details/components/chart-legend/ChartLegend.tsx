import './ChartLegend.css';

interface Props {
  label: string;
}

const ChartLegend = ({ label }: Props) => {
  return (
    <div className="chart-legend">{label}</div>
  )
}

export default ChartLegend