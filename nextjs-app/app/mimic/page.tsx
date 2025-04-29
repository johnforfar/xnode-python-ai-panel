import dynamic from "next/dynamic";

const Mimic = dynamic(() => import("@/components/mimic"), { ssr: false });

export default function MimicPage() {
  return <Mimic />;
}
